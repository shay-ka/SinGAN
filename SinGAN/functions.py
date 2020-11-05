import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch.nn as nn
import scipy.io as sio
import math
from skimage import io as img
from skimage import color, morphology, filters
#from skimage import morphology
#from skimage import filters
from SinGAN.imresize import imresize
import os
import random
from sklearn.cluster import KMeans
from scipy.io.wavfile import read
from scipy.io.wavfile import write
from scipy import signal
import math
# debug
from scipy.fft import fftshift
# debug

# custom weights initialization called on netG and netD

def read_image(opt):
    x = img.imread('%s%s' % (opt.input_img,opt.ref_image))
    return np2torch(x)

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def norm(x):
    out = (x -0.5) *2
    return out.clamp(-1, 1)

def norm_audio(x):
    if isinstance(x, torch.Tensor):
        x = torch.sign(x) * torch.sqrt(torch.abs(x)) / math.sqrt(32768)
    elif isinstance(x, np.ndarray):
        x = np.sign(x) * np.sqrt(np.abs(x)) / math.sqrt(32768)
    else:
        print('@ norm_log: unknown type!!! type(x)=', type(x))
    return x

def denorm_audio(x):
    if isinstance(x, torch.Tensor):
        x = torch.sign(x) * (x * math.sqrt(32768)) ** 2
    elif isinstance(x, np.ndarray):
        x = np.sign(x) * (x * math.sqrt(32768)) ** 2
    else:
        print('@ norm_log: unknown type!!! type(x)=', type(x))
    return x

#def denorm2image(I1,I2):
#    out = (I1-I1.mean())/(I1.max()-I1.min())
#    out = out*(I2.max()-I2.min())+I2.mean()
#    return out#.clamp(I2.min(), I2.max())

#def norm2image(I1,I2):
#    out = (I1-I2.mean())*2
#    return out#.clamp(I2.min(), I2.max())

def convert_image_np(inp):
    # print("@ convert_image_np: inp.shape = ", inp.shape)
    if inp.shape[1]==3:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1,:,:,:])
        inp = inp.numpy().transpose((1,2,0))
    else:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1,:,:])
        inp = inp.numpy().transpose((0,1))
        # mean = np.array([x/255.0 for x in [125.3,123.0,113.9]])
        # std = np.array([x/255.0 for x in [63.0,62.1,66.7]])

    inp = np.clip(inp,0,1)
    return inp

def convert_audio_np(inp, opt):
    # avoid norm / denorm (reverted)
    inp = denorm(inp)
    # print('@ convert_audio_np: max(inp)=', torch.max(inp), ' | min(inp)=', torch.min(inp))
    inp = move_to_cpu(inp[-1,-1,:])
    inp = inp.numpy() #.transpose((0,1))

    #inp = np.clip(inp,0,1)
    # convert to 16bit
    # avoid norm / denorm (reverted)
    if opt.audio_norm == True:
        inp = (inp * 2) - 1
        inp = denorm_audio(inp)
    else:
        inp = inp * 65535 - 32768
    # inp = inp * 32768
    inp = inp.astype(np.int16)
    return inp

def convert_spectrogram_np(inp, opt):
    # print("@ convert_image_np: inp.shape = ", inp.shape)
    print("inp.shape = ", inp.shape, " | inp = ", inp)
    inp = denorm(inp)
    print("inp.shape = ", inp.shape, " | inp = ", inp)
    inp = move_to_cpu(inp[-1, :, :, :])
    # inp = inp.numpy().transpose((1, 2, 0))
    # print('@ convert_spectrogram_np: inp.shape = ', inp.shape)
    inp = np.clip(inp,0,1)

    inp_mag = inp[0, :, :]
    inp_phase = inp[1, :, :]
    print("inp_mag.shape = ", inp_mag.shape, " | inp_mag = ", inp_mag)
    print("inp_phase.shape = ", inp_phase.shape, " | inp_phase = ", inp_phase)
    # return to real scale
    inp_mag = inp_mag * 255
    inp_phase = inp_phase * 255
    print("inp_mag.shape = ", inp_mag.shape, " | inp_mag = ", inp_mag)
    print("inp_phase.shape = ", inp_phase.shape, " | inp_phase = ", inp_phase)
    inp_mag = inp_mag / 192 * float(opt.max_mag)
    inp_phase = inp_phase / 255 * 2 * math.pi - math.pi
    # reconstruct
    # debug
    t = np.arange(inp_mag.shape[1]) # np.arange(inp_mag.shape[0] / 256) * 256 / 8000
    f = np.fft.fftfreq(inp_mag.shape[0]) * 8000
    plt.figure(1)
    plt.pcolormesh(t, fftshift(f), fftshift(inp_mag, axes=0), shading='gouraud')
    plt.title('inp_mag')
    # plt.figure(2)
    # plt.pcolormesh(t, fftshift(f), fftshift(inp_phase, axes=0), shading='gouraud')
    # plt.title('inp_phase')mi
    plt.show()
    print("min(inp_mag) = ", torch.min(inp_mag), " | max(inp_mag) = ", torch.max(inp_mag))
    print("min(inp_phase) = ", torch.min(inp_phase), " | max(inp_phase) = ", torch.max(inp_phase))
    # debug
    print("inp_mag.shape = ", inp_mag.shape, " | inp_mag = ", inp_mag)
    print("inp_phase.shape = ", inp_phase.shape, " | inp_phase = ", inp_phase)
    inp_fft_recon = np.multiply(np.sqrt(np.exp(inp_mag) - 1), np.exp(1j * inp_phase))
    print("inp_fft_recon.shape = ", inp_fft_recon.shape, " | inp_fft_recon = ", inp_fft_recon)
    inp_recon = np.array([], dtype=float)
    # print('@ convert_spectrogram_np: inp_mag.shape[0] = ', inp_mag.shape[0], ' | inp_fft_recon.shape = ', inp_fft_recon.shape)
    for i in range(inp_mag.shape[1]):
        inp_recon = np.concatenate((inp_recon, np.fft.ifft(inp_fft_recon[:, i])))
        # TODO: check why it's not perfectly reconstructed in to real values.
        inp_recon = np.abs(inp_recon)
    print("inp_recon.shape = ", inp_recon.shape, " | inp_recon = ", inp_recon)
    inp_recon = inp_recon.astype(np.int16)
    print("inp_recon.shape = ", inp_recon.shape, " | inp_recon = ", inp_recon)
    return inp_recon

def save_image(real_cpu,receptive_feild,ncs,epoch_num,file_name):
    fig,ax = plt.subplots(1)
    if ncs==1:
        ax.imshow(real_cpu.view(real_cpu.size(2),real_cpu.size(3)),cmap='gray')
    else:
        #ax.imshow(convert_image_np(real_cpu[0,:,:,:].cpu()))
        ax.imshow(convert_image_np(real_cpu.cpu()))
    rect = patches.Rectangle((0,0),receptive_feild,receptive_feild,linewidth=5,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    ax.axis('off')
    plt.savefig(file_name)
    plt.close(fig)

def convert_image_np_2d(inp):
    inp = denorm(inp)
    inp = inp.numpy()
    # mean = np.array([x/255.0 for x in [125.3,123.0,113.9]])
    # std = np.array([x/255.0 for x in [63.0,62.1,66.7]])
    # inp = std*
    return inp

def generate_noise(size,num_samp=1,device='cuda',type='gaussian', scale=1, input_type='image'):
    if type == 'gaussian':
        if len(size) == 3:
            noise = torch.randn(num_samp, size[0], round(size[1] / scale), round(size[2] / scale), device=device)
            noise = upsampling(noise, size[1], size[2])
        else:
            noise = torch.randn(1, num_samp, round(size[0] / scale), round(size[1] / scale), device=device)
            noise = upsampling(noise, size[0], size[1])
            noise = noise[0, :, :, :]
    if type == 'gaussian_mixture':
        if len(size) == 3:
            noise1 = torch.randn(num_samp, size[0], size[1], size[2], device=device) + 5
            noise2 = torch.randn(num_samp, size[0], size[1], size[2], device=device)
        else:
            noise1 = torch.randn(num_samp, size[0], size[1], device=device) + 5
            noise2 = torch.randn(num_samp, size[0], size[1], device=device)
        noise = noise1 + noise2
    if type == 'uniform':
        if len(size) == 3:
            noise = torch.randn(num_samp, size[0], size[1], size[2], device=device)
        else:
            noise = torch.randn(num_samp, size[0], size[1], device=device)
    return noise

def plot_learning_curves(G_loss,D_loss,epochs,label1,label2,name):
    fig,ax = plt.subplots(1)
    n = np.arange(0,epochs)
    plt.plot(n,G_loss,n,D_loss)
    #plt.title('loss')
    #plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend([label1,label2],loc='upper right')
    plt.savefig('%s.png' % name)
    plt.close(fig)

def plot_learning_curve(loss,epochs,name):
    fig,ax = plt.subplots(1)
    n = np.arange(0,epochs)
    plt.plot(n,loss)
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.savefig('%s.png' % name)
    plt.close(fig)

def upsampling(im,sx,sy):
    m = nn.Upsample(size=[round(sx),round(sy)],mode='bilinear',align_corners=True)
    return m(im)

def reset_grads(model,require_grad):
    for p in model.parameters():
        p.requires_grad_(require_grad)
    return model

def move_to_gpu(t):
    if (torch.cuda.is_available()):
        t = t.to(torch.device('cuda'))
    return t

def move_to_cpu(t):
    t = t.to(torch.device('cpu'))
    return t

def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA, device):
    #print real_data.size()
    alpha = torch.rand(1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)#cuda() #gpu) #if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)


    interpolates = interpolates.to(device)#.cuda()
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),#.cuda(), #if use_cuda else torch.ones(
                                  #disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    #LAMBDA = 1
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

def read_image(opt):
    if opt.conv_spectrogram == True:
        opt.sample_rate, x = read('%s/%s' % (opt.input_dir, opt.input_name))
        # align to 256
        x = x[:x.shape[0] - (x.shape[0] % 256)]
        # convert to spectrogram 256 x N
        # f, t, x = signal.spectrogram(x, opt.sample_rate)
        # assuming x length is 256 X N
        x_mag = np.array([], dtype=float)
        x_phase = np.array([], dtype=float)
        for i in range(int(x.shape[0]/256)):
            x_fft = np.fft.fft(x[256 * i : 256 * i + 256])
            # f = np.fft.fftfreq(x_fft.shape[0]) * opt.sample_rate
            x_mag = np.concatenate((x_mag.reshape(256, -1), np.log(np.abs(x_fft) ** 2 + 1).reshape(-1, 1)), axis=1)
            x_phase = np.concatenate((x_phase.reshape(256, -1), np.angle(x_fft).reshape(-1, 1)), axis=1)
        # strech x_mag, x_phase to (0-255)
        opt.max_mag = np.max(x_mag)
        x_mag = x_mag / opt.max_mag * 192 # strech to 3/4 of range
        x_phase = (x_phase + math.pi) / (2 * math.pi) * 255
        # stack magnitude and phase
        # print('@ read_image: x_mag.shape = ', x_mag.shape)
        # print('@ read_image: x_phase.shape = ', x_phase.shape)
        x = np.stack((x_mag, x_phase))
        # print('@ read_image: x.shape = ', x.shape)

    else:
        x = img.imread('%s/%s' % (opt.input_dir,opt.input_name))
        # print("@ read_image: x.shape = ", x.shape)
        # print("@ read_image: x = ", x)
    x = np2torch(x,opt)
    x = x[:,0:3,:,:]
    # print("@ read_image: x.shape = ", x.shape)
    # print("@ read_image: x = ", x)
    return x

def read_audio(opt):
    opt.sample_rate, x = read('%s/%s' % (opt.input_dir,opt.input_name))
    # print("@ read_image: x.shape = ", x.shape)
    # print("@ read_image: x = ", x)
    x = np.array(x, dtype=float)
    x = np2torch(x, opt)
    x = x.permute((0,2,1))
    # print("@ read_audio: x.shape = ", x.shape)
    # print("@ read_audio: x = ", x)
    #x = x[:, 0:1, :, :]
    return x

def read_image_dir(dir,opt):
    x = img.imread('%s' % (dir))
    x = np2torch(x,opt)
    x = x[:,0:3,:,:]
    return x

def np2torch(x,opt):
    if opt.input_type == 'image':
        if opt.conv_spectrogram == True:
            x = x[:, :, :, None]
            x = x.transpose((3, 0, 1, 2)) / 255
        else:
            if opt.nc_im == 3:
                x = x[:,:,:,None]
                x = x.transpose((3, 2, 0, 1))/255
                # print("@ read_image: x.shape = ", x.shape)
                # print("@ read_image: x = ", x)
            else:
                x = color.rgb2gray(x)
                x = x[:,:,None,None]
                x = x.transpose(3, 2, 0, 1)
    else:
        # aviod norm / denorm (reverted)
        if opt.audio_norm == True:
            x = norm_audio(x)
            x = (x + 1) / 2
        else:
            x = (x + 32768) / 65535 # audio file is normalized to [0,1]
        # print('@ np2torch: max(inp)=', x.max(), ' | min(inp)=', x.min())
        # x = x / 32768
        x = x[:, None, None]
        x = x.transpose(2, 0, 1)
        # print("@ read_image: x.shape = ", x.shape)
        # print("@ read_image: x = ", x)
    x = torch.from_numpy(x)
    if not(opt.not_cuda):
        x = move_to_gpu(x)
    x = x.type(torch.cuda.FloatTensor) if not(opt.not_cuda) else x.type(torch.FloatTensor)
    #x = x.type(torch.FloatTensor)
    # aviod norm / denorm (reverted)
    x = norm(x)
    # print('@ np2torch: max(inp)=', torch.max(x), ' | min(inp)=', torch.min(x))
    # if opt.input_type == 'image':
    #     x = norm(x)
    return x

def torch2uint8(x):
    x = x[0,:,:,:]
    x = x.permute((1,2,0))
    x = 255*denorm(x)
    x = x.cpu().numpy()
    x = x.astype(np.uint8)
    return x

def read_image2np(opt):
    x = img.imread('%s/%s' % (opt.input_dir,opt.input_name))
    x = x[:, :, 0:3]
    return x

def save_networks(netG,netD,z,opt):
    torch.save(netG.state_dict(), '%s/netG.pth' % (opt.outf))
    torch.save(netD.state_dict(), '%s/netD.pth' % (opt.outf))
    torch.save(z, '%s/z_opt.pth' % (opt.outf))

def adjust_scales2image(real_,opt):
    #opt.num_scales = int((math.log(math.pow(opt.min_size / (real_.shape[2]), 1), opt.scale_factor_init))) + 1
    opt.num_scales = math.ceil((math.log(math.pow(opt.min_size / (min(real_.shape[2], real_.shape[3])), 1), opt.scale_factor_init))) + 1
    scale2stop = math.ceil(math.log(min([opt.max_size, max([real_.shape[2], real_.shape[3]])]) / max([real_.shape[2], real_.shape[3]]),opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    opt.scale1 = min(opt.max_size / max([real_.shape[2], real_.shape[3]]),1)  # min(250/max([real_.shape[0],real_.shape[1]]),1)
    real = imresize(real_, opt.scale1, opt)
    #opt.scale_factor = math.pow(opt.min_size / (real.shape[2]), 1 / (opt.stop_scale))
    opt.scale_factor = math.pow(opt.min_size/(min(real.shape[2],real.shape[3])),1/(opt.stop_scale))
    scale2stop = math.ceil(math.log(min([opt.max_size, max([real_.shape[2], real_.shape[3]])]) / max([real_.shape[2], real_.shape[3]]),opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    return real

def adjust_scales2image_SR(real_,opt):
    print("@ adjust_scales2image_SR: real_.shape = ", real_.shape)
    if opt.input_type == 'image':
        opt.min_size = 18
        opt.num_scales = int((math.log(opt.min_size / min(real_.shape[2], real_.shape[3]), opt.scale_factor_init))) + 1
        print("@ adjust_scales2image_SR: opt.num_scales = ", opt.num_scales)
        scale2stop = int(math.log(min(opt.max_size , max(real_.shape[2], real_.shape[3])) / max(real_.shape[0], real_.shape[3]), opt.scale_factor_init))
        print("@ adjust_scales2image_SR: scale2stop = ", scale2stop)
        opt.stop_scale = opt.num_scales - scale2stop
        print("@ adjust_scales2image_SR: opt.stop_scale = ", opt.stop_scale)
        opt.scale1 = min(opt.max_size / max([real_.shape[2], real_.shape[3]]), 1)  # min(250/max([real_.shape[0],real_.shape[1]]),1)
        print("@ adjust_scales2image_SR: opt.scale1 = ", opt.scale1)
        real = imresize(real_, opt.scale1, opt)
        print("@ adjust_scales2image_SR: real.shape = ", real.shape)
        #opt.scale_factor = math.pow(opt.min_size / (real.shape[2]), 1 / (opt.stop_scale))
        opt.scale_factor = math.pow(opt.min_size/(min(real.shape[2],real.shape[3])),1/(opt.stop_scale))
        print("@ adjust_scales2image_SR: opt.scale_factor = ", opt.scale_factor)
        scale2stop = int(math.log(min(opt.max_size, max(real_.shape[2], real_.shape[3])) / max(real_.shape[0], real_.shape[3]), opt.scale_factor_init))
        print("@ adjust_scales2image_SR: scale2stop = ", scale2stop)
        opt.stop_scale = opt.num_scales - scale2stop
        print("@ adjust_scales2image_SR: opt.stop_scale = ", opt.stop_scale)
    else:
        opt.num_scales = int((math.log(opt.min_size / real_.shape[2], opt.scale_factor_init))) + 1
        print("@ adjust_scales2image_SR: opt.num_scales = ", opt.num_scales)
        scale2stop = int(math.log(min(opt.max_size, real_.shape[2]) / real_.shape[2], opt.scale_factor_init))
        print("@ adjust_scales2image_SR: scale2stop = ", scale2stop)
        opt.stop_scale = opt.num_scales - scale2stop
        print("@ adjust_scales2image_SR: opt.stop_scale = ", opt.stop_scale)
        opt.scale1 = min(opt.max_size / real_.shape[2], 1)  # min(250/max([real_.shape[0],real_.shape[1]]),1)
        print("@ adjust_scales2image_SR: opt.scale1 = ", opt.scale1)
        real = imresize(real_, opt.scale1, opt)
        print("@ adjust_scales2image_SR: real.shape = ", real.shape)
        # opt.scale_factor = math.pow(opt.min_size / (real.shape[2]), 1 / (opt.stop_scale))
        opt.scale_factor = math.pow(opt.min_size / (real.shape[2]), 1 / (opt.stop_scale))
        print("@ adjust_scales2image_SR: opt.scale_factor = ", opt.scale_factor)
        scale2stop = int(math.log(min(opt.max_size, real_.shape[2]) / real_.shape[2], opt.scale_factor_init))
        print("@ adjust_scales2image_SR: scale2stop = ", scale2stop)
        opt.stop_scale = opt.num_scales - scale2stop
        print("@ adjust_scales2image_SR: opt.stop_scale = ", opt.stop_scale)
    return real

def creat_reals_pyramid(real,reals,opt):
    if opt.input_type == 'image':
        real = real[:,0:3,:,:]
    for i in range(0,opt.stop_scale+1,1):
        scale = math.pow(opt.scale_factor,opt.stop_scale-i)
        curr_real = imresize(real,scale,opt)
        # print('@ creat_reals_pyramid: curr_real.shape = ', curr_real.shape)
        reals.append(curr_real)
    return reals


def load_trained_pyramid(opt, mode_='train'):
    #dir = 'TrainedModels/%s/scale_factor=%f' % (opt.input_name[:-4], opt.scale_factor_init)
    mode = opt.mode
    opt.mode = 'train'
    if (mode == 'animation_train') | (mode == 'SR_train') | (mode == 'paint_train'):
        opt.mode = mode
    dir = generate_dir2save(opt)
    if(os.path.exists(dir)):
        Gs = torch.load('%s/Gs.pth' % dir)
        Zs = torch.load('%s/Zs.pth' % dir)
        reals = torch.load('%s/reals.pth' % dir)
        NoiseAmp = torch.load('%s/NoiseAmp.pth' % dir)
    else:
        print('no appropriate trained model is exist, please train first')
    opt.mode = mode
    return Gs,Zs,reals,NoiseAmp

def generate_in2coarsest(reals,scale_v,scale_h,opt):
    real = reals[opt.gen_start_scale]
    real_down = upsampling(real, scale_v * real.shape[2], scale_h * real.shape[3])
    if opt.gen_start_scale == 0:
        in_s = torch.full(real_down.shape, 0, device=opt.device)
    else: #if n!=0
        in_s = upsampling(real_down, real_down.shape[2], real_down.shape[3])
    return in_s

def generate_dir2save(opt):
    dir2save = None
    if (opt.mode == 'train') | (opt.mode == 'SR_train'):
        dir2save = 'TrainedModels/%s/scale_factor=%f,alpha=%d' % (opt.input_name[:-4], opt.scale_factor_init,opt.alpha)
    elif (opt.mode == 'animation_train') :
        dir2save = 'TrainedModels/%s/scale_factor=%f_noise_padding' % (opt.input_name[:-4], opt.scale_factor_init)
    elif (opt.mode == 'paint_train') :
        dir2save = 'TrainedModels/%s/scale_factor=%f_paint/start_scale=%d' % (opt.input_name[:-4], opt.scale_factor_init,opt.paint_start_scale)
    elif opt.mode == 'random_samples':
        dir2save = '%s/RandomSamples/%s/gen_start_scale=%d' % (opt.out,opt.input_name[:-4], opt.gen_start_scale)
    elif opt.mode == 'random_samples_arbitrary_sizes':
        dir2save = '%s/RandomSamples_ArbitrerySizes/%s/scale_v=%f_scale_h=%f' % (opt.out,opt.input_name[:-4], opt.scale_v, opt.scale_h)
    elif opt.mode == 'animation':
        dir2save = '%s/Animation/%s' % (opt.out, opt.input_name[:-4])
    elif opt.mode == 'SR':
        dir2save = '%s/SR/%s' % (opt.out, opt.sr_factor)
    elif opt.mode == 'harmonization':
        dir2save = '%s/Harmonization/%s/%s_out' % (opt.out, opt.input_name[:-4],opt.ref_name[:-4])
    elif opt.mode == 'editing':
        dir2save = '%s/Editing/%s/%s_out' % (opt.out, opt.input_name[:-4],opt.ref_name[:-4])
    elif opt.mode == 'paint2image':
        dir2save = '%s/Paint2image/%s/%s_out' % (opt.out, opt.input_name[:-4],opt.ref_name[:-4])
        if opt.quantization_flag:
            dir2save = '%s_quantized' % dir2save
    return dir2save

def post_config(opt):
    # init fixed parameters
    opt.device = torch.device("cpu" if opt.not_cuda else "cuda:0")
    opt.niter_init = opt.niter
    opt.noise_amp_init = opt.noise_amp
    opt.nfc_init = opt.nfc
    opt.min_nfc_init = opt.min_nfc
    opt.scale_factor_init = opt.scale_factor
    opt.out_ = 'TrainedModels/%s/scale_factor=%f/' % (opt.input_name[:-4], opt.scale_factor)
    if opt.mode == 'SR':
        opt.alpha = 100

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if torch.cuda.is_available() and opt.not_cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    return opt

def calc_init_scale(opt):
    in_scale = math.pow(1/2,1/3)
    iter_num = round(math.log(1 / opt.sr_factor, in_scale))
    in_scale = pow(opt.sr_factor, 1 / iter_num)
    return in_scale,iter_num

def quant(prev,device):
    arr = prev.reshape((-1, 3)).cpu()
    kmeans = KMeans(n_clusters=5, random_state=0).fit(arr)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    x = centers[labels]
    x = torch.from_numpy(x)
    x = move_to_gpu(x)
    x = x.type(torch.cuda.FloatTensor) if () else x.type(torch.FloatTensor)
    #x = x.type(torch.FloatTensor.to(device))
    x = x.view(prev.shape)
    return x,centers

def quant2centers(paint, centers):
    arr = paint.reshape((-1, 3)).cpu()
    kmeans = KMeans(n_clusters=5, init=centers, n_init=1).fit(arr)
    labels = kmeans.labels_
    #centers = kmeans.cluster_centers_
    x = centers[labels]
    x = torch.from_numpy(x)
    x = move_to_gpu(x)
    x = x.type(torch.cuda.FloatTensor) if torch.cuda.is_available() else x.type(torch.FloatTensor)
    #x = x.type(torch.cuda.FloatTensor)
    x = x.view(paint.shape)
    return x

    return paint


def dilate_mask(mask,opt):
    if opt.mode == "harmonization":
        element = morphology.disk(radius=7)
    if opt.mode == "editing":
        element = morphology.disk(radius=20)
    mask = torch2uint8(mask)
    mask = mask[:,:,0]
    mask = morphology.binary_dilation(mask,selem=element)
    mask = filters.gaussian(mask, sigma=5)
    nc_im = opt.nc_im
    opt.nc_im = 1
    mask = np2torch(mask,opt)
    opt.nc_im = nc_im
    mask = mask.expand(1, 3, mask.shape[2], mask.shape[3])
    plt.imsave('%s/%s_mask_dilated.png' % (opt.ref_dir, opt.ref_name[:-4]), convert_image_np(mask), vmin=0,vmax=1)
    mask = (mask-mask.min())/(mask.max()-mask.min())
    return mask


