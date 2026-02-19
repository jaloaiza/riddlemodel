using UnityEngine;
using LLama;
using LLama.Common;
using LLama.Native;
using System.Threading.Tasks;
using System.IO;

public class LlamaTest : MonoBehaviour
{
    private string modelPath = "Assets/Models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";
    private string response = "";
    private bool inferenceComplete = false;
    
    void Start()
    {
        Debug.Log("Testing LLamaSharp with inference...");
        
        if (!System.IO.File.Exists(modelPath))
        {
            Debug.LogError($"Model file not found at: {modelPath}");
            return;
        }
        
        Debug.Log($"Model file found at: {modelPath}");
        
        // Set library search path as environment variable
        string pluginsPath = Path.Combine(Application.dataPath, "Plugins/win-x64");
        Debug.Log($"Adding to library path: {pluginsPath}");
        
        // Add to DYLD_LIBRARY_PATH for macOS
        string currentPath = System.Environment.GetEnvironmentVariable("DYLD_LIBRARY_PATH") ?? "";
        System.Environment.SetEnvironmentVariable("DYLD_LIBRARY_PATH", 
            string.IsNullOrEmpty(currentPath) ? pluginsPath : $"{pluginsPath}:{currentPath}");
        
        try
        {
            // Test initialization
            _ = NativeApi.llama_max_devices();
            Debug.Log("LLamaSharp native library initialized successfully!");
            
            Task.Run(() => LoadAndInfer());
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Failed to initialize: {e.Message}");
            if (e.InnerException != null)
            {
                Debug.LogError($"Inner: {e.InnerException.Message}");
            }
        }
    }
    
    private async Task LoadAndInfer()
    {
        try
        {
            Debug.Log("Loading model...");
            
            var parameters = new ModelParams(modelPath)
            {
                ContextSize = 2048,
                GpuLayerCount = 0,
                Threads = 4
            };
            
            using var model = LLamaWeights.LoadFromFile(parameters);
            Debug.Log("Model loaded!");
            
            using var context = model.CreateContext(parameters);
            var executor = new InteractiveExecutor(context);
            
            var prompt = "What is 2+2? Answer briefly.";
            Debug.Log($"Prompt: {prompt}");
            Debug.Log("Generating response...");
            
            response = "";
            await foreach (var text in executor.InferAsync(prompt, new InferenceParams()
            {
                MaxTokens = 50,
            }))
            {
                response += text;
            }
            
            inferenceComplete = true;
            
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Error: {e.Message}");
            Debug.LogError($"Stack: {e.StackTrace}");
        }
    }
    
    void Update()
    {
        if (inferenceComplete)
        {
            Debug.Log($"Response: {response}");
            inferenceComplete = false;
        }
    }
}