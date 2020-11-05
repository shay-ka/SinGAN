from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
from SinGAN.imresize import imresize
import SinGAN.functions as functions
# import soundfile as sf
# from pypesq import pesq
# from scipy.io.wavfile import read

if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='training image name', default="33039_LR.png")#required=True)
    parser.add_argument('--sr_factor', help='super resolution factor', type=float, default=4)
    parser.add_argument('--mode', help='task to be done', default='SR')
    parser.add_argument('--input_type', help='input image or audio', default='image')
    parser.add_argument('--sample_rate', help='input image or audio', default=None)
    parser.add_argument('--conv_spectrogram', help='convert audio to spectrorgam', default=False)
    parser.add_argument('--max_mag', help='maximum magnitude value', default=0.0)
    parser.add_argument('--audio_norm', help='normalize audio with costume func', type=bool, default=False)
    opt = parser.parse_args()
    opt = functions.post_config(opt)
    opt.conv_spectrogram = bool(opt.conv_spectrogram)
    print('opt.conv_spectrogram = ', opt.conv_spectrogram, ' | type = ', type(opt.conv_spectrogram))
    print('opt.max_mag = ', opt.max_mag, ' | type = ', type(opt.max_mag))
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)
    if dir2save is None:
        print('task does not exist')
    #elif (os.path.exists(dir2save)):
    #    print("output already exist")
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass

        mode = opt.mode
        in_scale, iter_num = functions.calc_init_scale(opt)
        opt.scale_factor = 1 / in_scale
        opt.scale_factor_init = 1 / in_scale
        opt.mode = 'train'
        dir2trained_model = functions.generate_dir2save(opt)
        if (os.path.exists(dir2trained_model)):
            Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)
            opt.mode = mode
        else:
            print('*** Train SinGAN for SR ***')
            if opt.input_type == 'image':
                real = functions.read_image(opt)
                opt.min_size = 18
            else:
                # input_type == 'audio'
                real = functions.read_audio(opt)
            real = functions.adjust_scales2image_SR(real, opt)
            train(opt, Gs, Zs, reals, NoiseAmp)
            opt.mode = mode
        print('%f' % pow(in_scale, iter_num))
        Zs_sr = []
        reals_sr = []
        NoiseAmp_sr = []
        Gs_sr = []
        real = reals[-1]  # read_image(opt)
        real_ = real
        opt.scale_factor = 1 / in_scale
        opt.scale_factor_init = 1 / in_scale
        for j in range(1, iter_num + 1, 1):
            real_ = imresize(real_, pow(1 / opt.scale_factor, 1), opt)
            reals_sr.append(real_)
            Gs_sr.append(Gs[-1])
            NoiseAmp_sr.append(NoiseAmp[-1])
            z_opt = torch.full(real_.shape, 0, device=opt.device)
            if opt.input_type == 'image':
                m = nn.ZeroPad2d(5)
            else:
                temp_pad1 = ((opt.ker_size - 1) * opt.num_layer) / 2
                m = nn.ConstantPad1d(int(temp_pad1), 0)
            z_opt = m(z_opt)
            Zs_sr.append(z_opt)
        out = SinGAN_generate(Gs_sr, Zs_sr, reals_sr, NoiseAmp_sr, opt, in_s=reals_sr[0], num_samples=1)
        if opt.input_type == 'image':
            out = out[:, :, 0:int(opt.sr_factor * reals[-1].shape[2]), 0:int(opt.sr_factor * reals[-1].shape[3])]
        else:
            out = out[:, :, 0:int(opt.sr_factor * reals[-1].shape[2])]
        dir2save = functions.generate_dir2save(opt)
        if opt.input_type == 'image':
            if opt.conv_spectrogram == True:
                write('%s/%s_HR.wav' % (dir2save, opt.input_name[:-4]), int(opt.sample_rate) * int(opt.sr_factor), functions.convert_spectrogram_np(out.detach(), opt))
                # write('%s/%s_HR.wav' % (dir2save, opt.input_name[:-4]), int(opt.sample_rate) ,functions.convert_spectrogram_np(out.detach(), opt))
            else:
                plt.imsave('%s/%s_HR.png' % (dir2save,opt.input_name[:-4]), functions.convert_image_np(out.detach()), vmin=0, vmax=1)
        else:
            print("@ main: type(opt.sample_rate)=",type(opt.sample_rate),"opt.sample_rate=",opt.sample_rate)
            # print("@ main: functions.convert_audio_np(out.detach())=",functions.convert_audio_np(out.detach()),"type=",type(functions.convert_audio_np(out.detach())))
            write('%s/%s_HR.wav' % (dir2save,opt.input_name[:-4]), int(opt.sample_rate) * int(opt.sr_factor), functions.convert_audio_np(out.detach(), opt))
            # # calculate pesq of upscaling by interpulation and by SinGan
            # ref_name = opt.input_name.replace('%dkHz' % (int(int(opt.sample_rate) / 1000)), '%dkHz' % (int(int(opt.sample_rate) * 2 / 1000)))
            # print('%s/%s' % (opt.input_dir, ref_name))
            # # sr, ref = read('%s/%s' % (opt.input_dir, ref_name))
            # ref, sr = sf.read('%s/%s' % (opt.input_dir, ref_name))
            # print('sr=',sr,'ref - ', ref.shape, '\n', ref)
            # interp_name = opt.input_name.replace('%dkHz' % (int(int(opt.sample_rate) / 1000)), '%dkHz_interp' % (int(int(opt.sample_rate) * 2 / 1000)))
            # # sr, interp = read('%s/%s' % (opt.input_dir,interp_name))
            # interp, sr = sf.read('%s/%s' % (opt.input_dir, interp_name))
            # print('sr=', sr, 'interp - ', interp.shape, '\n', interp)
            # # sr, sgOut = read('%s/%s_HR.wav' % (dir2save,opt.input_name[:-4]))
            # sgOut, sr = sf.read('%s/%s_HR.wav' % (dir2save, opt.input_name[:-4]))
            # print('sr=',sr,'sgOut - ', sgOut.shape,'\n',sgOut)
            # # trying with normalization
            # # ref = (ref / 32768)
            # # interp = (interp / 32768)
            # # sgOut = (sgOut / 32768)
            # score_interp = pesq(ref, interp, sr)
            # score_sg = pesq(ref, sgOut, sr)
            # print(score_interp, "  |  ", score_sg)
            # print('PESQ Score: interuplation - %f | SinGan - %f, ' % (score_interp, score_sg))
            # print('PESQ score: ref - ', pesq(ref, ref, sr))
            # # calc MSE
            # mse_interp = sum(abs(ref - interp) ** 2) / len(ref)
            # mse_sg = sum(abs(ref - sgOut) ** 2) / len(ref)
            # print('Avg MSE: interuplation - %f | SinGan - %f, ' % (mse_interp, mse_sg))


