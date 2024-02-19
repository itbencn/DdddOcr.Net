using Microsoft.ML.OnnxRuntime.Tensors;

using System.Text;

namespace DdddOcr.Net
{
    public class DDDDOCR
    {
        DdddOcrMode Mode { get; }
        DdddOcrOptions Options { get; }
        SessionOptions SessionOptions { get; }
        InferenceSession InferenceSession { get; }

        #region Basic
        public DDDDOCR(DdddOcrMode mode, bool use_gpu = false, int device_id = 0)
        {
#if DEBUG
            Console.WriteLine($"欢迎使用ddddocr，本项目专注带动行业内卷");
            Console.WriteLine($"python版开发作者：https://github.com/sml2h3/ddddocr");
            Console.WriteLine($"C#/NET版移植作者：https://github.com/itbencn/DdddOcr.Net");
            Console.WriteLine($"本项目仅作为移植项目未经过大量测试 生产环境谨慎使用");
            Console.WriteLine($"请勿违反所在地区法律法规 合理合法使用本项目");
#endif
            if (!Enum.IsDefined(mode))
            {
                throw new NotSupportedException($"不支持的模式:{mode}");
            }
            Mode = mode;
            Options = new DdddOcrOptions();
            var onnx_path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, Mode.GetDescription());
            if (!File.Exists(onnx_path))
            {
                throw new FileNotFoundException($"{mode}模式对应的模型文件不存在:{onnx_path}");
            }
            Options.Charset = Mode switch
            {
                DdddOcrMode.ClassifyOld => Global.OCR_OLD_CHARSET,
                DdddOcrMode.ClassifyBeta => Global.OCR_BETA_CHARSET,
                _ => [],
            };
            SessionOptions = new SessionOptions();
            if (use_gpu)
            {
                SessionOptions.AppendExecutionProvider_CUDA(device_id);
            }
            else
            {
                SessionOptions.AppendExecutionProvider_CPU();
            }
            InferenceSession = new InferenceSession(File.ReadAllBytes(onnx_path), SessionOptions);
        }

        public DDDDOCR(string import_onnx_path, string charsets_path, bool use_gpu = false, int device_id = 0)
        {
#if DEBUG
            Console.WriteLine($"欢迎使用ddddocr，本项目专注带动行业内卷");
            Console.WriteLine($"python版开发作者：https://github.com/sml2h3/ddddocr");
            Console.WriteLine($"C#/NET版移植作者：https://github.com/itbencn/DdddOcr.Net");
            Console.WriteLine($"请合理合法使用本项目 本项目未经过大量测试 生产环境谨慎使用");
#endif
            Mode = DdddOcrMode.Import;
            if (!File.Exists(import_onnx_path))
            {
                throw new FileNotFoundException($"文件不存在:{import_onnx_path}");
            }
            if (!File.Exists(charsets_path))
            {
                throw new FileNotFoundException($"文件不存在:{charsets_path}");
            }
            Options = DdddOcrOptions.FromJsonFile(charsets_path);
            if (Options == null)
            {
                throw new FileLoadException("数据格式错误");
            }
            SessionOptions = new SessionOptions();
            if (use_gpu)
            {
                SessionOptions.AppendExecutionProvider_CUDA(device_id);
            }
            else
            {
                SessionOptions.AppendExecutionProvider_CPU();
            }
            InferenceSession = new InferenceSession(File.ReadAllBytes(import_onnx_path), SessionOptions);
        }

        ~DDDDOCR()
        {
            Dispose();
        }

        public void Dispose()
        {
            SessionOptions?.Dispose();
            InferenceSession?.Dispose();
            GC.SuppressFinalize(this);
        }
        #endregion

        #region classification
        public string Classify(byte[] bytes, bool pngFix = false)
        {
            if (Mode == DdddOcrMode.Detect)
            {
                throw new InvalidOperationException("当前识别类型为目标检测");
            }
            using var image = Mat.FromImageData(bytes, ImreadModes.AnyColor);
            if (image.Width == 0 && image.Height == 0)
            {
                throw new InvalidOperationException("载入图像数据损坏或图片类型错误");
            }
            var inputs = ClassifyPrepareProcessing(image);
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs = InferenceSession.Run(inputs);
            var predictions = outputs.First(x => x.Name == "output").Value as DenseTensor<long>;
            if (predictions == null)
            {
                return string.Empty;
            }
            var result = new StringBuilder();
            foreach (long prediction in predictions)
            {
                result.Append(Options.Charset[(int)prediction]);
            }
            return result.ToString();
        }

        static readonly float[] mean = [0.485f, 0.456f, 0.406f];
        static readonly float[] std = [0.229f, 0.224f, 0.225f];

        List<NamedOnnxValue> ClassifyPrepareProcessing(Mat image, bool pngFix = false)
        {
            #region resize
            Mat resizedImg;

            if (Mode == DdddOcrMode.Import)
            {
                if (Options.Resize.Width == -1)
                {
                    if (Options.Word)
                    {
                        resizedImg = image.Resize(new Size(Options.Resize.Height, Options.Resize.Height), interpolation: InterpolationFlags.Linear);
                    }
                    else
                    {
                        resizedImg = image.Resize(new Size(image.Width * Convert.ToDouble(Options.Resize.Height / image.Height), Options.Resize.Height), interpolation: InterpolationFlags.Linear);
                    }
                }
                else
                {
                    resizedImg = image.Resize(new Size(Options.Resize.Width, Options.Resize.Height), interpolation: InterpolationFlags.Linear);
                }
                if (Options.Channel == 1)
                {
                    //BGR2GRAY? RGB2GRAY?
                    resizedImg = image.CvtColor(ColorConversionCodes.BGR2GRAY);
                }
                else
                {
                    if (pngFix)
                    {
                        resizedImg = PngRgbaToRgbWhiteBackground(image);
                    }
                    else
                    {
                        //resizedImg = image.convert('RGB')
                    }
                }
            }
            else
            {
                //BGR2GRAY? RGB2GRAY?
                resizedImg = image.Resize(new Size(image.Width * Convert.ToDouble(64d / image.Height), 64d), interpolation: InterpolationFlags.Linear).CvtColor(ColorConversionCodes.BGR2GRAY);
            }
            #endregion

            #region tensor
            int channels = resizedImg.Channels();
            var tensor = new DenseTensor<float>([1, channels, resizedImg.Height, resizedImg.Width]);
            for (int y = 0; y < resizedImg.Height; y++)
            {
                for (int x = 0; x < resizedImg.Width; x++)
                {
                    if (Mode == DdddOcrMode.Import)
                    {
                        if (Options.Channel == 1 || channels == 1)
                        {
                            byte color = resizedImg.At<byte>(y, x);
                            tensor[0, 0, y, x] = ((color / 255f) - 0.456f) / 0.224f;
                        }
                        else
                        {
                            Vec3b color = resizedImg.At<Vec3b>(y, x);
                            for (int c = 0; c < channels; c++)
                            {
                                tensor[0, c, y, x] = ((color[c] / 255f) - mean[c]) / std[c];
                            }
                        }
                    }
                    else
                    {
                        byte color = resizedImg.At<byte>(y, x);
                        tensor[0, 0, y, x] = ((color / 255f) - 0.5f) / 0.5f;
                    }
                }
            }
            resizedImg.Dispose();
            return [NamedOnnxValue.CreateFromTensor("input1", tensor)];
            #endregion
        }

        static Mat PngRgbaToRgbWhiteBackground(Mat src)
        {
            if (src.Channels() != 4)
            {
                return src;
            }
            var whiteBackground = new Mat(src.Size(), MatType.CV_8UC3, Scalar.White);
            var srcChannels = Cv2.Split(src);
            using Mat alphaChannel = srcChannels[3];
            using var rgb = new Mat();
            Cv2.Merge([srcChannels[0], srcChannels[1], srcChannels[2]], rgb);
            rgb.CopyTo(whiteBackground, alphaChannel);
            foreach (var mat in srcChannels)
            {
                mat.Dispose();
            }
            return whiteBackground;
        }
        #endregion

        #region detection
        public int Detect(byte[] bytes)
        {
            if (Mode != DdddOcrMode.Detect)
            {
                throw new InvalidOperationException("当前识别类型为文字识别");
            }
            using var image = Mat.FromImageData(bytes, ImreadModes.AnyColor);
            if (image.Width == 0 && image.Height == 0)
            {
                throw new InvalidOperationException("载入图像数据损坏或图片类型错误");
            }
            var inputSize = new Size(416, 416);
            var inputs = DetectPrepareProcessing(image, inputSize);
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs = InferenceSession.Run(inputs);
            var predictions = outputs.First(x => x.Name == "output").Value as DenseTensor<float>;

            //......

            return 0;
        }

        List<NamedOnnxValue> DetectPrepareProcessing(Mat image, Size inputSize, List<int>? swap = default)
        {
            #region resize
            swap ??= [2, 0, 1];
            Mat paddedImg;

            if (image.Channels() == 3)
            {
                paddedImg = new Mat(inputSize, MatType.CV_8UC3, new Scalar(114, 114, 114));
            }
            else
            {
                paddedImg = new Mat(inputSize, MatType.CV_8UC1, new Scalar(114));
            }

            float ratio = Math.Min((float)inputSize.Height / image.Rows, (float)inputSize.Width / image.Cols);
            var resizedImg = new Mat();
            Cv2.Resize(image, resizedImg, new Size((image.Cols * ratio), (image.Rows * ratio)), 0, 0, InterpolationFlags.Linear);

            resizedImg.CopyTo(paddedImg[new Rect(0, 0, resizedImg.Cols, resizedImg.Rows)]);

            resizedImg = new Mat();
            if (swap.Count == 3)
            {
                using var paddedImgFloat = new Mat();
                paddedImg.ConvertTo(paddedImgFloat, MatType.CV_32F);
                Mat[] srcChannels = Cv2.Split(paddedImgFloat);
                Cv2.Merge([srcChannels[swap[0]], srcChannels[swap[1]], srcChannels[swap[2]]], resizedImg);
            }
            else
            {
                paddedImg.ConvertTo(resizedImg, MatType.CV_32F);
            }
            paddedImg.Dispose();
            #endregion

            #region tensor
            int channels = resizedImg.Channels();
            var tensor = new DenseTensor<float>([1, channels, inputSize.Height, inputSize.Height]);
            for (int y = 0; y < resizedImg.Height; y++)
            {
                for (int x = 0; x < resizedImg.Width; x++)
                {
                    Vec3b color = image.At<Vec3b>(y, x);
                    for (int c = 0; c < channels; c++)
                    {
                        tensor[0, c, y, x] = color[c];
                    }
                }
            }
            resizedImg.Dispose();
            return [NamedOnnxValue.CreateFromTensor("images", tensor)];
            #endregion
        }
        #endregion

        #region slide

        #endregion
    }
}
