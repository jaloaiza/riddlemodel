using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using TMPro;
using UnityEngine;
using UnityEngine.UI;
using System.Threading.Tasks;
using LLama;
using LLama.Common;
using LLama.Native;
using LLama.Sampling;
using LLama.Transformers;

public class RiddleGameUI : MonoBehaviour
{
    [Header("UI References")]
    public TextMeshProUGUI riddleText;
    public TMP_InputField answerInput;
    public TextMeshProUGUI feedbackText;
    [SerializeField] private GameObject _feedbackGO;
    [SerializeField] private GameObject _riddleGO;
    public Button submitButton;
    public Button newRiddleButton;
    
    [Header("Riddle Model Fields")]
    [SerializeField] private float _temp = 0.9f;
    [SerializeField] private float _topP = 0.9f;
    [SerializeField] private int _topK = 40;
    
    [Header("Grade Model Fields")]
    [SerializeField] private float _gTemp = 0.4f;
    [SerializeField] private float _gTopP = 0.9f;
    [SerializeField] private int _gTopK = 40;
    
    private LLamaWeights _model;
    
    private LLamaContext _riddleContext;
    private LLamaContext _gradeContext;
    
    private InteractiveExecutor _riddleExecutor;
    private InteractiveExecutor _gradeExecutor;
    
    private ChatSession _riddleSession;
    private ChatSession _gradeSession;
    
    private string _systemPrompt;
    private string _gradeSystemPrompt;

    private string _currentRiddle = "";
    private string _currentAnswer = "";

    private ChatHistory _baseHistory;

    async void Start()
    {
        _feedbackGO.SetActive(false);
        submitButton.interactable = false;
        newRiddleButton.interactable = false;
        riddleText.text = "Loading model...";

        await InitializeModel();

        newRiddleButton.interactable = true;
        submitButton.onClick.AddListener(OnSubmitAnswer);
        newRiddleButton.onClick.AddListener(OnNewRiddle);

        // Auto-generate first riddle
        await GenerateRiddle();
    }

    async Task InitializeModel()
    {
        // tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
        string modelPath = Path.Combine(Application.streamingAssetsPath, "Models", "gemma-3-1b-it-Q4_K_M.gguf");
        
        var parameters = new ModelParams(modelPath)
        {
            ContextSize = 1024,
            GpuLayerCount = 4
        };

        // Load Model
        _model = await LLamaWeights.LoadFromFileAsync(parameters);
        
        // Create Contexts
        _riddleContext = _model.CreateContext(parameters);
        _gradeContext = _model.CreateContext(parameters);
        
        // Create Executors
        _riddleExecutor = new InteractiveExecutor(_riddleContext);
        _gradeExecutor = new InteractiveExecutor(_gradeContext);

        _systemPrompt =
            "Create a simple riddle.\n" +
            "Return EXACTLY this format:\n" +
            "RIDDLE: <1-3 sentences>\n" +
            "ANSWER: <1-4 words>\n" +
            "<END_RIDDLE>";

        _gradeSystemPrompt =
            "You are a strict grader.\n" +
            "DO NOT repeat the riddle.\n" +
            "DO NOT restate the answers.\n" +
            "Reply with EXACTLY one line:\n";

        var riddleHistory = new ChatHistory();
        riddleHistory.AddMessage(AuthorRole.System, _systemPrompt);
        
        var gradeHistory = new ChatHistory();
        gradeHistory.AddMessage(AuthorRole.System, _gradeSystemPrompt);

        _riddleSession = new ChatSession(_riddleExecutor, riddleHistory);
        _gradeSession = new ChatSession(_gradeExecutor, gradeHistory);
        
        _riddleSession.HistoryTransform = new PromptTemplateTransformer(_model, withAssistant: true);
        _gradeSession.HistoryTransform  = new PromptTemplateTransformer(_model, withAssistant: true);
    }

    async Task GenerateRiddle()
    {
        riddleText.text = "Generating riddle...";
        _riddleGO.SetActive(true);
        _feedbackGO.SetActive(false);
        submitButton.interactable = false;
        feedbackText.text = "";
        answerInput.text = "";

        _riddleSession.History.Messages.Clear();
        _riddleSession.History.AddMessage(AuthorRole.System, _systemPrompt);
        
        var ip = new InferenceParams
        {
            MaxTokens = 120,
            SamplingPipeline = new DefaultSamplingPipeline
            {
              Temperature = _temp,
              TopP = _topP,
              TopK = _topK,
              RepeatPenalty = 1.15f,
              PenaltyCount = 128,
              PenalizeNewline = false,
              Seed = (uint) UnityEngine.Random.Range(1, int.MaxValue)
            },
            
            AntiPrompts = new List<string> { "<END_RIDDLE>", "\nUser:", "\nAssistant:" }
        };
        
        string variation = $" (id:{UnityEngine.Random.Range(0, 1000000)})";
        string response = "";
        await foreach (var token in _riddleSession.ChatAsync(
                           new ChatHistory.Message(AuthorRole.User, "Generate one simple riddle." + variation),
                           ip))
        {
            response += token;
        }
        
        string cleaned = response.Replace("<END_RIDDLE>", "");
        cleaned = Sanitize(cleaned);

        // Parse
        _currentRiddle = ExtractBetween(cleaned, "RIDDLE:", "ANSWER:").Trim();
        _currentAnswer = ExtractAfter(cleaned, "ANSWER:").Trim();

        // Fallback: if parsing fails, treat whole thing as riddle
        if (string.IsNullOrWhiteSpace(_currentRiddle))
        {
            _currentRiddle = cleaned.Trim();
            _currentAnswer = "";
        }

        riddleText.text = _currentRiddle;
        submitButton.interactable = true;
    }
    

    async void OnSubmitAnswer()
    {
        string userAnswer = answerInput.text.Trim();
        if (string.IsNullOrEmpty(userAnswer)) return;
        
        _gradeSession.History.Messages.Clear();
        _gradeSession.History.AddMessage(AuthorRole.System, _gradeSystemPrompt);
        
        _feedbackGO.SetActive(true);
        _riddleGO.SetActive(false);
        
        submitButton.interactable = false;
        newRiddleButton.interactable = false;
        feedbackText.text = "Evaluating...";

        string prompt =
            $"Riddle: {_currentRiddle}\n" +
            $"Correct answer: {_currentAnswer}\n" +
            $"User answer: {userAnswer}\n" +
            "Reply following this format: <Correct or Incorrect> - <one short sentence reason>";


        var gradeIp = new InferenceParams
        {
            MaxTokens = 200,
            SamplingPipeline = new DefaultSamplingPipeline
            {
                Temperature = _gTemp,
                TopP = _gTopP,
                TopK = _gTopK,
                RepeatPenalty = 1.05f,
                PenaltyCount = 128,
                Seed = (uint)UnityEngine.Random.Range(1, int.MaxValue)
            },
            AntiPrompts = new List<string>
            {
                "<end_of_turn>",   // gemma end marker
                "\nUser:", "User:" // safety
            }

        };

        string response = "";
        try
        {
            await foreach (var token in _gradeSession.ChatAsync(
                               new ChatHistory.Message(AuthorRole.User, prompt),
                               gradeIp))
            {
                response += token;
            }
        }
        catch (Exception e)
        {
            Debug.LogException(e);
            feedbackText.text = "Error during evaluation. Check console.";
            newRiddleButton.interactable = true;
            submitButton.interactable = true;
            return;
        }

        Debug.Log("Response: " + response);
        Debug.Log("Grade response chars: " + response.Length);

        // Clean up and display
        string cleaned = response
            .Replace("<start_of_turn>", "")
            .Replace("<end_of_turn>", "");
        cleaned = Sanitize(cleaned);

        feedbackText.text = cleaned;
        newRiddleButton.interactable = true;
        submitButton.interactable = true;
    }
    
    async void OnNewRiddle()
    {
        await GenerateRiddle();
    }

    void OnDestroy()
    {
        _riddleSession = null;
        _gradeSession = null;
        
        _riddleContext?.Dispose();
        _gradeContext?.Dispose();
        _model?.Dispose();

    }
    
    // =================== HELPER FUNCTIONS ================
    static string Sanitize(string s)
    {
        if (string.IsNullOrEmpty(s)) return s;

        // Normalize newlines
        s = s.Replace("\r\n", "\n").Replace("\r", "\n");

        // Remove control chars except \n and \t
        var sb = new System.Text.StringBuilder(s.Length);
        foreach (char c in s)
        {
            if (c == '\n' || c == '\t' || !char.IsControl(c))
                sb.Append(c);
        }
        s = sb.ToString();

        // Collapse excessive whitespace/newlines
        s = System.Text.RegularExpressions.Regex.Replace(s, @"[ \t]+\n", "\n");
        s = System.Text.RegularExpressions.Regex.Replace(s, @"\n{3,}", "\n\n");
        s = System.Text.RegularExpressions.Regex.Replace(s, @"[ \t]{2,}", " ");

        return s.Trim();
    }
    
    static string StripAtFirst(string s, string marker)
    {
        int idx = s.IndexOf(marker, StringComparison.Ordinal);
        return idx >= 0 ? s.Substring(0, idx) : s;
    }
    
    static string ExtractBetween(string s, string a, string b)
    {
        int ia = s.IndexOf(a, StringComparison.OrdinalIgnoreCase);
        if (ia < 0) return "";
        ia += a.Length;
        int ib = s.IndexOf(b, ia, StringComparison.OrdinalIgnoreCase);
        if (ib < 0) return s.Substring(ia);
        return s.Substring(ia, ib - ia);
    }

    static string ExtractAfter(string s, string a)
    {
        int ia = s.IndexOf(a, StringComparison.OrdinalIgnoreCase);
        if (ia < 0) return "";
        return s.Substring(ia + a.Length);
    }


}
