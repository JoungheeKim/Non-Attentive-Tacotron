import hydra
import os
import torch
from tqdm import tqdm, trange
from tacotron.utils import reset_logging, set_seed, get_abspath, ResultWriter
from tacotron import get_process, get_model, get_vocgan
from tacotron.configs import NonAttentiveTacotronConfig
from hydra.core.config_store import ConfigStore
import logging
from transformers import (
    get_linear_schedule_with_warmup,
)
from tacotron.vocgan_generator import Generator
import soundfile as sf

def init():
    cs = ConfigStore.instance()

    ## base

    cs.store(group="base", name='non_taco', node=NonAttentiveTacotronConfig)


@hydra.main(config_path=os.path.join(".", "configs"), config_name="train")
def main(cfg):

    ## Resent Logging
    reset_logging()

    args = cfg.base

    ## GPU setting
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    set_seed(args.seed)

    ## load_dataset
    processor = get_process(args)(args)
    tokenizer = processor.build_tokenizer()

    train_dataset = processor.get_dataset(tokenizer, split='train')
    valid_dataset = processor.get_dataset(tokenizer, split='valid')

    args.num_labels = tokenizer.get_num_labels()
    args.num_special_labels = tokenizer.get_num_special_labels()

    model = get_model(args)(args)
    model.to(args.device)

    generator = None
    generator_path = get_abspath(args.generator_path)
    if os.path.exists(generator_path):
        generator = get_vocgan(generator_path)
        generator.to(args.device)
        generator.eval()

    ## test
    # train_dataloader = train_dataset.load_dataloader(
    #      shuffle=False, batch_size=2
    # )
    # batch = train_dataset.__getitem__(2)
    # batch = {key: (item.to(args.device) if type(item) == torch.Tensor else item) for key, item in batch.items()}
    # print(batch)
    #
    # print(tokenizer.decode(batch['input_ids'], batch['special_input_ids']))
    #
    # print("shape", batch['mel_specs'].shape)
    # print(batch['mel_specs'])
    #
    # audio = generator.generate_audio(batch['mel_specs'])
    # sf.write('sample1.wav', audio, args.sampling_rate, 'PCM_24')
    #
    # for batch in train_dataloader:
    #     print('confirm1', tokenizer.decode(batch['input_ids'][0], batch['special_input_ids'][0]))
    #     print('confirm2', tokenizer.decode(batch['input_ids'][1], batch['special_input_ids'][1]))
    #
    #     batch = {key: (item.to(args.device) if type(item) == torch.Tensor else item) for key, item in batch.items()}
    #
    #     #print(batch['durations'])
    #     #print(batch['input_ids'])
    #     mel_specs = batch['mel_specs'].squeeze()
    #     print("shape", mel_specs.shape)
    #     print(mel_specs)
    #
    #     #net_output = model(**batch)
    #
    #     ## generate audio
    #     audio = generator.generate_audio(mel_specs[0])
    #     ## save audio
    #     sf.write('sample1.wav', audio, args.sampling_rate, 'PCM_24')
    #
    #
    #     #loss = model.get_loss(**net_output, **batch)
    #     #print('loss', loss)
    #     break
    #
    ## train model
    writer = ResultWriter(args.experiments_path)
    results = {}

    ## training
    train_results = train(args, train_dataset, valid_dataset, model, tokenizer, generator)
    results.update(**train_results)
    writer.update(args, **results)




def train(args, train_dataset, valid_dataset, model, tokenizer, generator=None):
    logging.info("start training")

    ## load dataloader
    train_dataloader = train_dataset.load_dataloader(
        shuffle=True, batch_size=args.train_batch_size
    )

    if args.max_steps > 0:
        t_total = args.max_steps

        args.num_train_epochs = (
            args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        )
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=args.weight_decay)
    args.warmup_steps = int(args.warmup_percent * t_total)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    logging.info("  Num Epochs = %d", args.num_train_epochs)
    logging.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logging.info("  Total optimization steps = %d", t_total)
    logging.info("  Train Batch size = %d", args.train_batch_size)
    logging.info("  Train Data size = %d", len(train_dataset))

    step = 0
    global_step = 0

    best_loss = 1e10
    best_loss_step = 0

    stop_iter = False

    #train_iterator = trange(0, int(args.num_train_epochs), desc="Epoch")
    model.zero_grad()
    for epoch_idx in range(0, int(args.num_train_epochs)):

        ## load dataloader
        train_dataloader = train_dataset.load_dataloader(
            shuffle=True, batch_size=args.train_batch_size
        )

        for batch in train_dataloader:
            step += 1
            model.train()

            batch = {key: (item.to(args.device) if type(item) == torch.Tensor else item) for key, item in batch.items()}
            net_output = model(**batch)
            loss = model.get_loss(**net_output, **batch)
            final_loss = loss['loss']

            if args.gradient_accumulation_steps > 1:
                final_loss = final_loss / args.gradient_accumulation_steps
            if args.n_gpu > 1:
                final_loss = final_loss.mean()
            if args.fp16:
                with amp.scale_loss(final_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                final_loss.backward()

            #train_iterator.set_postfix_str(s="loss = {:.8f}".format(float(final_loss)), refresh=True)
            if step % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.grad_clip_thresh)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_thresh)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                ## logging
                if (args.logging_steps > 0 and global_step % args.logging_steps == 0 and global_step > 0):
                    logging.info("***** epoch [{}] loss [{:.4}] b_mse [{:.4}] b_mae [{:.4}] a_mse [{:.4}] a_mae [{:.4}] duration [{:.4}] *****".format(
                        str(epoch_idx), 
                        loss['loss'].detach().cpu().item(),
                        loss['predicted_mse'].detach().cpu().item(),
                        loss['predicted_mae'].detach().cpu().item(),
                        loss['predicted_postnet_mse'].detach().cpu().item(),
                        loss['predicted_postnet_mae'].detach().cpu().item(),
                        loss['predicted_durations_mse'].detach().cpu().item(),
                        )
                    )

                if (args.steps_per_evaluate > 0 and global_step % args.steps_per_evaluate == 0 and global_step > 0):

                    ## audio prepare path
                    audio_save_path = os.path.join(args.save_path, 'audio', str(global_step))
                    audio_save_path = get_abspath(audio_save_path)
                    os.makedirs(audio_save_path, exist_ok=True)

                    # if (args.logging_steps > 0 and global_step % args.logging_steps == 0):
                    results = evaluate(args, valid_dataset, model, generator, audio_save_path)
                    eval_loss = results['loss']
                    if eval_loss < best_loss:
                        best_loss = eval_loss
                        best_loss_step = global_step

                    logging.info("***** best_loss : %.4f *****", best_loss)

                if (args.steps_per_checkpoint > 0 and global_step % args.steps_per_checkpoint == 0 and global_step > 0):
                    model_save_path = os.path.join(args.save_path, 'model', str(global_step))
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(model_save_path)
                    tokenizer.save_pretrained(model_save_path)

            if args.max_steps > 0 and global_step > args.max_steps:
                stop_iter = True
                break

        if stop_iter:
            break

    return {'best_valid_loss': best_loss,
            'best_valid_loss_step': best_loss_step,
            }



def evaluate(args, test_dataset, model, generator=None, save_path=''):

    ## load dataloader
    test_dataloader = test_dataset.load_dataloader(
        shuffle=False, batch_size=1
    )

    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    logging.info("***** Running evaluation *****")
    eval_loss = 0.0
    nb_eval_steps = 0

    model.eval()
    for batch in tqdm(test_dataloader, desc="Evaluating"):

        batch = {key: (item.to(args.device) if type(item) == torch.Tensor else item) for key, item in batch.items()}

        with torch.no_grad():
            net_output = model(**batch)
            loss = model.get_loss(**net_output, **batch)
            eval_loss += loss['loss'].cpu().item()

            if generator is not None:
                net_output = model.inference(**batch)
                ## generate audio
                audio = generator.generate_audio(**net_output)

                ## save audio
                sf.write(os.path.join(save_path, '{}.wav'.format(str(nb_eval_steps))), audio, args.sampling_rate, 'PCM_24')



        nb_eval_steps += 1

    eval_loss = eval_loss/nb_eval_steps

    results = {
        'loss' : eval_loss,
    }

    logging.info("  %s = %s", 'loss', str(results['loss']))
    model.train()

    return results
























if __name__ == "__main__":
    init()
    main()
