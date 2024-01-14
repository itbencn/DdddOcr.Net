using DdddOcr.Net;

namespace DdddOcr.Demo
{
    internal class Program
    {
        static void Main(string[] args)
        {
            DDDDOCR ddddOcr = new DDDDOCR(DdddOcrMode.ClassifyOld);

            var result = ddddOcr.Classify(File.ReadAllBytes(@"code1.png"));

            Console.WriteLine(result);

            Console.ReadKey();
        }
    }
}
