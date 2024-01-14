using System.Text;
using System.Text.Json;

namespace DdddOcr.Net
{
    public class DdddOcrOptions
    {
        public List<string> Charset { get; set; } = [];
        public bool Word { get; set; } = false;
        public Size Resize { get; set; }
        public int Channel { get; set; } = 1;

        public string ToJson()
        {
            JsonSerializerOptions jsonOptions = new JsonSerializerOptions();
            jsonOptions.Converters.Add(new SizeJsonConverter());
            return JsonSerializer.Serialize(this, jsonOptions);
        }

        public static DdddOcrOptions? FromJson(string json)
        {
            JsonSerializerOptions jsonOptions = new JsonSerializerOptions();
            jsonOptions.Converters.Add(new SizeJsonConverter());
            return JsonSerializer.Deserialize<DdddOcrOptions>(json, jsonOptions);
        }

        public static DdddOcrOptions? FromJsonFile(string path)
        {
            return FromJson(File.ReadAllText(path, Encoding.UTF8));
        }
    }
}
