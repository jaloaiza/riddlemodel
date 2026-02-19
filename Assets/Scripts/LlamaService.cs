using System;
using System.Collections.Concurrent;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using LLama;
using LLama.Common;
using LLama.Native;
using UnityEngine;

public sealed class LlamaService : MonoBehaviour
{
    public static LlamaService Instance { get; private set; }

    [Header("Model")]
    [Tooltip("Put your gguf in StreamingAssets for builds.")]
    public string ggufFileName = "Models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";

    [Header("Runtime")]
    [SerializeField] private uint contextSize = 2048;
    [SerializeField] private int gpuLayers = 0;
    [SerializeField] private int threads = 4;

    private LLamaWeights _weights;
    private LLamaContext _context;
    private InteractiveExecutor _executor;

    private readonly ConcurrentQueue<Action> _mainThreadActions = new();

    public bool IsReady => _executor != null;

    void Awake()
    {
        if (Instance != null) { Destroy(gameObject); return; }
        Instance = this;
        DontDestroyOnLoad(gameObject);
    }

    async void Start()
    {
        try
        {
            BootstrapNativeLibraryPath();

            // Smoke-test native lib
            _ = NativeApi.llama_max_devices();

            await InitializeAsync();
            Debug.Log("LlamaService ready.");
        }
        catch (Exception e)
        {
            Debug.LogError($"Llama init failed: {e.Message}\n{e}");
        }
    }

    void Update()
    {
        while (_mainThreadActions.TryDequeue(out var a))
            a?.Invoke();
    }

    private void BootstrapNativeLibraryPath()
    {
        string pluginsPath = Path.Combine(Application.dataPath, "Plugins", "win-x64");

        // Windows
        var path = Environment.GetEnvironmentVariable("PATH") ?? "";
        if (!path.Contains(pluginsPath))
            Environment.SetEnvironmentVariable("PATH", pluginsPath + ";" + path);

        // Linux
        var ld = Environment.GetEnvironmentVariable("LD_LIBRARY_PATH") ?? "";
        if (!ld.Contains(pluginsPath))
            Environment.SetEnvironmentVariable("LD_LIBRARY_PATH", pluginsPath + ":" + ld);

        // macOS
        var dyld = Environment.GetEnvironmentVariable("DYLD_LIBRARY_PATH") ?? "";
        if (!dyld.Contains(pluginsPath))
            Environment.SetEnvironmentVariable("DYLD_LIBRARY_PATH", pluginsPath + ":" + dyld);
    }

    private Task InitializeAsync()
    {
        return Task.Run(() =>
        {
            var modelPath = Path.Combine(Application.streamingAssetsPath, ggufFileName);

            if (!File.Exists(modelPath))
                throw new FileNotFoundException($"Model not found: {modelPath}");

            var p = new ModelParams(modelPath)
            {
                ContextSize = (uint)contextSize,
                GpuLayerCount = gpuLayers,
                Threads = threads
            };

            _weights = LLamaWeights.LoadFromFile(p);
            _context = _weights.CreateContext(p);
            _executor = new InteractiveExecutor(_context);
        });
    }

    /// <summary>
    /// Streams tokens via onToken. Safe to update Unity UI inside onToken (it is invoked on main thread).
    /// </summary>
    public async Task<string> GenerateAsync(
        string prompt,
        Action<string> onToken = null,
        CancellationToken ct = default,
        int maxTokens = 128)
    {
        
        if (!IsReady) throw new InvalidOperationException("LlamaService not initialized yet.");

        var full = "";

        // Run inference on a worker thread, but marshal token callbacks to main thread.
        await Task.Run(async () =>
        {
            await foreach (var token in _executor.InferAsync(prompt, new InferenceParams
            {
                MaxTokens = maxTokens
            }))
            {
                ct.ThrowIfCancellationRequested();

                full += token;

                if (onToken != null)
                {
                    var captured = token;
                    _mainThreadActions.Enqueue(() => onToken(captured));
                }
            }
        }, ct);

        return full;
    }

    void OnDestroy()
    {
        try { _context?.Dispose(); } catch { }
        try { _weights?.Dispose(); } catch { }
        if (Instance == this) Instance = null;
    }
}
