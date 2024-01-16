using DdddOcr.Net;

namespace DdddOcr.Demo
{
    internal class Program
    {
        static void Main(string[] args)
        {
            var image = File.ReadAllBytes(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "code.gif"));
            var type = image.GetImageFormat();
            if (type.FileExtensions.Contains("gif"))
            {
                image = image.GifToPng();
            }
            DDDDOCR ddddOcr = new DDDDOCR(DdddOcrMode.ClassifyBeta);

            var result = ddddOcr.Classify(image);

            Console.WriteLine(result.ToLower());

            Console.ReadKey();
        }
    }
}
