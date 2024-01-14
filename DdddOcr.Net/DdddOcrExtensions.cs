using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace DdddOcr.Net
{
    static class DdddOcrExtensions
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
    }
}
