using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats;

namespace DdddOcr.Net
{
    public static partial class DdddOcrExtensions
    {
        public static string GetDescription(this Enum value)
        {
            Type type = value.GetType();
            string? name = Enum.GetName(type, value);
            FieldInfo? field = type.GetField(name);
            DescriptionAttribute? attribute = field?.GetCustomAttribute<DescriptionAttribute>();
            return attribute != null ? attribute.Description : name;
        }

        private static void EnsureMode(this DDDDOCR predictor, DdddOcrMode mode)
        {
            if (Enum.IsDefined(mode))
                throw new InvalidOperationException($"The mode does not support {mode}");
        }

        public static void ThrowIfNull<T>(T argument, string paramName, [CallerMemberName] string methodName = "")
        {
            if (argument is null)
            {
                throw new ArgumentNullException(paramName, $"{methodName} => Parameter '{paramName}' cannot be null.");
            }
        }

        /// <summary>
        /// 判断图片类型
        /// </summary>
        /// <param name="imageBytes"></param>
        /// <returns></returns>
        public static IImageFormat GetImageFormat(this byte[] imageBytes)
        {
            return Image.DetectFormat(imageBytes);
        }

        /// <summary>
        /// GifToPng
        /// </summary>
        /// <param name="gifBytes"></param>
        /// <param name="index">zero-based index</param>
        /// <returns></returns>
        public static byte[] GifToPng(this byte[] gifBytes, int index = 0)
        {
            using var image = Image.Load(gifBytes);
            using var ms = new MemoryStream();
            image.SaveAsPng(ms);
            return ms.ToArray();
        }
    }
}
