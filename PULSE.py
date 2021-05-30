import os

from attribute_detector import load_attribute_detector_from_checkpoint
from bicubic import BicubicDownsampleTargetSize
from stylegan import G_synthesis, G_mapping
from dataclasses import dataclass
from SphericalOptimizer import SphericalOptimizer
from pathlib import Path
import numpy as np
import time
import torch
from loss import LossBuilder
from functools import partial
from drive import open_url


class PULSE(torch.nn.Module):
    def __init__(self, cache_dir, face_comparer_config, verbose=True, use_stylegan2=False):
        super(PULSE, self).__init__()

        cuda_id = os.environ['CUDA_VISIBLE_DEVICES'].split(',')[0]

        # if use_stylegan2:
        #     if verbose: print("Loading Synthesis Network (StyleGan2)")
        #     self.synthesis = Generator(1024, 512, 8, channel_multiplier=2).cuda(1)
        #     checkpoint = torch.load('stylegan2_pytorch/stylegan2-ffhq-config-f.pt')
        #     self.synthesis.load_state_dict(checkpoint["g_ema"])
        #     self.generate_on_device_2 = True
        # else:
        if verbose: print("Loading Synthesis Network")
        self.synthesis = G_synthesis().cuda(f'cuda:{cuda_id}')
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open_url("https://drive.google.com/uc?id=1TCViX1YpQyRsklTVYEJwdbmK91vklCo8", cache_dir=cache_dir,
                      verbose=verbose) as f:
            self.synthesis.load_state_dict(torch.load(f))
        self.generate_on_device_2 = False
        self.verbose = verbose



        for param in self.synthesis.parameters():
            param.requires_grad = False

        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2)

        if Path("gaussian_fit.pt").exists():
            # slight hack to get it from this flag
            cuda_id = os.environ['CUDA_VISIBLE_DEVICES'].split(',')[0]
            self.gaussian_fit = torch.load("gaussian_fit.pt", map_location={'cuda:0': f'cuda:{cuda_id}'})
        else:
            if self.verbose: print("\tLoading Mapping Network")
            mapping = G_mapping().cuda()

            with open_url("https://drive.google.com/uc?id=14R6iHGf5iuVx3DMNsACAl7eBr7Vdpd0k", cache_dir=cache_dir, verbose=verbose) as f:
                    mapping.load_state_dict(torch.load(f))

            if self.verbose: print("\tRunning Mapping Network")
            with torch.no_grad():
                torch.manual_seed(0)
                latent = torch.randn((1000000,512),dtype=torch.float32, device="cuda")
                latent_out = torch.nn.LeakyReLU(5)(mapping(latent))
                self.gaussian_fit = {"mean": latent_out.mean(0), "std": latent_out.std(0)}
                torch.save(self.gaussian_fit,"gaussian_fit.pt")
                if self.verbose: print("\tSaved \"gaussian_fit.pt\"")

        from train_face_comparer import load_face_comparer_module
        # Create an inception resnet (in eval mode):
        net, trainer = load_face_comparer_module(face_comparer_config, for_eval=True)
        self.face_features_extractor = net.face_comparer.cuda()

        attribute_detector_ckpt = face_comparer_config + '.attribs.ckpt'
        if os.path.exists(attribute_detector_ckpt):
            self.attribute_detector = load_attribute_detector_from_checkpoint(attribute_detector_ckpt)
            self.attribute_detector.eval()
        else:
            self.attribute_detector = None

        if hasattr(self.face_features_extractor.face_features_extractor, 'race_detector') and \
                self.face_features_extractor.face_features_extractor.race_detector:
            print(f"Using fairface race detector as attribute detector")
            self.attribute_detector = self.face_features_extractor.face_features_extractor.race_detector

        if hasattr(self.face_features_extractor.face_features_extractor, 'attr_detector') and \
                self.face_features_extractor.face_features_extractor.attr_detector:
            print(f"Using CelebA Attr detector as attribute detector")
            self.attribute_detector = self.face_features_extractor.face_features_extractor.attr_detector

        if not self.attribute_detector:
            print(f"Warning: no attribute detector checkpoint found. Attribute loss terms will crash.")

    def forward(self, ref_im,
                target_identity_im,
                seed,
                loss_str,
                eps,
                noise_type,
                num_trainable_noise_layers,
                tile_latent,
                bad_noise_layers,
                opt_name,
                learning_rate,
                steps,
                lr_schedule,
                save_intermediate,
                **kwargs):

        if seed:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True

        batch_size = ref_im.shape[0]

        # Generate latent tensor
        if 'latent' in kwargs:
            latent = kwargs['latent']
        elif(tile_latent):
            latent = torch.randn(
                (batch_size, 1, 512), dtype=torch.float, requires_grad=True, device='cuda')
        else:
            latent = torch.randn(
                (batch_size, 18, 512), dtype=torch.float, requires_grad=True, device='cuda')

        # Generate list of noise tensors
        noise = [] # stores all of the noise tensors
        noise_vars = []  # stores the noise tensors that we want to optimize on

        for i in range(18):
            # dimension of the ith noise tensor
            initial_noise_tensor_exp = 3 if self.generate_on_device_2 else 2
            i_delta = 1 if self.generate_on_device_2 else 0
            res = (batch_size, 1, 2**((i+i_delta)//2+2), 2**((i+i_delta)//2+2))

            if(noise_type == 'zero' or i in [int(layer) for layer in bad_noise_layers.split('.')]):
                new_noise = torch.zeros(res, dtype=torch.float, device='cuda')
                new_noise.requires_grad = False
            elif(noise_type == 'fixed'):
                new_noise = torch.randn(res, dtype=torch.float, device='cuda')
                new_noise.requires_grad = False
            elif (noise_type == 'trainable'):
                new_noise = torch.randn(res, dtype=torch.float, device='cuda')
                if (i < num_trainable_noise_layers):
                    new_noise.requires_grad = True
                    noise_vars.append(new_noise)
                else:
                    new_noise.requires_grad = False
            else:
                raise Exception("unknown noise type")

            noise.append(new_noise)

        var_list = [latent]+noise_vars

        opt_dict = {
            'sgd': torch.optim.SGD,
            'adam': torch.optim.Adam,
            'sgdm': partial(torch.optim.SGD, momentum=0.9),
            'adamax': torch.optim.Adamax
        }
        opt_func = opt_dict[opt_name]
        opt = SphericalOptimizer(opt_func, var_list, lr=learning_rate)

        schedule_dict = {
            'fixed': lambda x: 1,
            'linear1cycle': lambda x: (9*(1-np.abs(x/steps-1/2)*2)+1)/10,
            'linear1cycledrop': lambda x: (9*(1-np.abs(x/(0.9*steps)-1/2)*2)+1)/10 if x < 0.9*steps else 1/10 + (x-0.9*steps)/(0.1*steps)*(1/1000-1/10),
        }
        schedule_func = schedule_dict[lr_schedule]
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt.opt, schedule_func)

        #target_identity_vector = None
        #if target_identity_im is not None:
        #    target_identity_vector = self.face_features_extractor.extract_features(target_identity_im)
        #    target_identity_vector = target_identity_vector.detach()

        loss_builder = LossBuilder(ref_im, target_identity_im, self.face_features_extractor, self.attribute_detector, loss_str, eps).cuda()

        min_loss = np.inf
        min_l2 = np.inf
        min_attr_loss = np.inf
        best_summary = ""
        start_t = time.time()
        gen_im = None


        if self.verbose: print("Optimizing")
        for j in range(steps):
            opt.opt.zero_grad()

            # Duplicate latent in case tile_latent = True
            if (tile_latent):
                latent_in = latent.expand(-1, 18, -1)
            else:
                latent_in = latent

            # Apply learned linear mapping to match latent distribution to that of the mapping network
            latent_in = self.lrelu(latent_in*self.gaussian_fit["std"] + self.gaussian_fit["mean"])

            # Normalize image to [0,1] instead of [-1,1]
            if self.generate_on_device_2:
                latent_in = latent_in.cuda(1)
                noise = [n.cuda(1) for n in noise]
                print(latent_in.device, self.synthesis.input.input.device)
                gen_im = (self.synthesis(latent_in, noise=noise) + 1) / 2
                latent_in = latent_in.cuda(0)
                gen_im = gen_im.cuda(0)
                noise = [n.cuda(0) for n in noise]
            else:
                gen_im = (self.synthesis(latent_in, noise) + 1) / 2

            # gen_identity_vector = self.face_features_extractor.forward(gen_im)
            # Calculate Losses
            loss, loss_dict = loss_builder(latent_in, gen_im)
            loss_dict['TOTAL'] = loss

            # Save best summary for log
            if(loss < min_loss):
                min_loss = loss
                best_summary = f'BEST ({j+1}) | '+' | '.join(
                [f'{x}: {y:.4f}' for x, y in loss_dict.items()])
                best_im = gen_im.clone()

            loss_l2 = loss_dict['L2']

            if(loss_l2 < min_l2):
                min_l2 = loss_l2
                iter_attr_loss = max([0] + [v for k, v in loss_dict.items() if k.startswith('ATTR')])
                min_attr_loss = iter_attr_loss

            # Save intermediate HR and LR images
            if(save_intermediate):
                yield (best_im.cpu().detach().clamp(0, 1),loss_builder.D(best_im).cpu().detach().clamp(0, 1))
            # print(loss.item())
            loss.backward()
            opt.step()
            scheduler.step()

        total_t = time.time()-start_t
        current_info = f' | time: {total_t:.1f} | it/s: {(j+1)/total_t:.2f} | batchsize: {batch_size}'
        if self.verbose: print(best_summary+current_info)
        attr_eps = 0.1 if 'latent' not in kwargs else 1
        if(min_l2 <= eps):
            if min_attr_loss != np.inf and min_attr_loss < attr_eps:
                hr = gen_im.clone().cpu().detach().clamp(0, 1)
                yield (hr,loss_builder.D(best_im).cpu().detach().clamp(0, 1))
                if 'loss_str_2' in kwargs and kwargs['loss_str_2']:
                    print("Reversing!")
                    hr_clone = hr.cuda()
                    hr_reversed, lr_reversed = next(self.forward(hr_clone, target_identity_im, seed,
                                                                 kwargs['loss_str_2'], eps, noise_type,
                                                                 num_trainable_noise_layers, tile_latent,
                                                                 bad_noise_layers, opt_name, learning_rate, steps,
                                                                 lr_schedule, save_intermediate, latent=latent))
                    yield hr_reversed, lr_reversed
            else:
                print(f"Could not find a face that matches attributes correctly within epsilon ({min_attr_loss:.4f} > {attr_eps})")
        else:
            print(f"Could not find a face that downscales correctly within epsilon ({min_l2:.4f} > {eps})")
