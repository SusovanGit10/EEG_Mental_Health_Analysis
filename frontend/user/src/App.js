import React, { useState } from "react";
import axios from "axios";
import {
  Container, Card, CardContent, Typography, Button, Box, Grid,
  Chip, Divider, Stack, Paper, CircularProgress, ThemeProvider, createTheme, CssBaseline
} from "@mui/material";
import {
  CloudUpload,
  Psychology,
  Memory,
  Biotech,
  Insights,
  Assessment,
  WorkspacePremium,
  ErrorOutlineOutlined
} from "@mui/icons-material";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer,
} from "recharts";

// --- Advanced UI Theme ---
const theme = createTheme({
  palette: {
    background: { default: "#f4f7f9", paper: "#ffffff" },
    primary: { main: "#2563eb" },
    text: { primary: "#1e293b", secondary: "#64748b" },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h4: { fontWeight: 700, letterSpacing: "-0.02em" },
    h6: { fontWeight: 600 },
  },
  shape: { borderRadius: 16 },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          boxShadow: "0 8px 24px rgba(0,0,0,0.06)",
          border: "1px solid #e2e8f0",
        }
      }
    },
    MuiButton: {
      styleOverrides: {
        root: { textTransform: "none", fontWeight: 600, borderRadius: 8 }
      }
    }
  }
});

// --- Dynamic Model Configuration ---
const MODEL_CONFIGS = [
  { key: "random_forest", title: "Random Forest", color: "#2e7d32", icon: <Biotech />, bg: "#e8f5e9", border: "#c8e6c9" },
  { key: "svm", title: "SVM", color: "#1565c0", icon: <Insights />, bg: "#e3f2fd", border: "#bbdefb" },
  { key: "cnn", title: "CNN", color: "#4527a0", icon: <Memory />, bg: "#ede7f6", border: "#d1c4e9" },
  { key: "xgboost", title: "XGBoost", color: "#ef6c00", icon: <Assessment />, bg: "#fff3e0", border: "#ffe0b2" },

  // ✅ NEW MODEL
  { key: "snn", title: "SNN", color: "#d81b60", icon: <Psychology />, bg: "#fce4ec", border: "#f8bbd0" }
];

export default function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await axios.post("https://susovanhuggingface10-eeg-mental-health-analysis.hf.space/predict", formData);
      setResult(res.data);
    } catch (error) {
      setResult({ error: "Failed to communicate with the server. Please check your backend." });
    } finally {
      setLoading(false);
    }
  };

  const getChartData = (waves) => [
    { name: "Alpha", value: waves?.alpha || 0 },
    { name: "Beta", value: waves?.beta || 0 },
    { name: "Gamma", value: waves?.gamma || 0 },
    { name: "Theta", value: waves?.theta || 0 }
  ];
  const labelMap = {
    "0": "Relaxed",
    "1": "Neutral",
    "2": "Concentrating"
  };

  const getComparisonData = () => {
    if (!result) return [];

    return MODEL_CONFIGS.map(config => {
      const model = result[config.key];

      const confidence = Number(model?.confidence || 0);
      const accuracy = Number(model?.accuracy || 0);

      const score = (confidence * 0.6) + (accuracy * 0.4);

      return {
        name: config.title,
        confidence,
        accuracy,
        score,
        prediction: labelMap[model?.prediction] || model?.prediction,
        color: config.color
      };
    });
  };
  const comparisonData = getComparisonData();

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Container maxWidth="xl" sx={{ py: 6, minHeight: '100vh' }}>

        {/* Header Section */}
        <Box textAlign="center" mb={6}>
          <Typography
            variant="h3"
            fontWeight="bold"
            gutterBottom
            sx={{
              background: "linear-gradient(90deg, #3f51b5, #00bcd4)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              gap: 2
            }}
          >
            <Psychology fontSize="inherit" sx={{ color: "#3f51b5", WebkitTextFillColor: "initial" }} />
            EEG Mental Health Analysis
          </Typography>
          <Typography variant="subtitle1" color="textSecondary" align="center" sx={{ textAlign: "center" }}>
            Upload your EEG data file to compare AI predictions instantly.
          </Typography>
        </Box>

        {/* Upload Dropzone */}
        <Paper
          elevation={0}
          sx={{
            p: 4,
            textAlign: "center",
            bgcolor: file ? "#f0f9ff" : "#f8f9fa",
            border: "2px dashed",
            borderColor: file ? "primary.main" : "#e0e0e0",
            transition: "all 0.3s ease",
            mb: 5,
            maxWidth: 600,
            mx: "auto"
          }}
        >
          <Stack direction="row" spacing={2} justifyContent="center" alignItems="center">
            <Button variant="outlined" component="label" size="large" sx={{ borderRadius: 2 }}>
              <CloudUpload sx={{ mr: 1 }} />
              {file ? file.name : "Choose File"}
              <input type="file" hidden onChange={(e) => setFile(e.target.files[0])} />
            </Button>
            <Button
              variant="contained"
              onClick={handleUpload}
              size="large"
              disableElevation
              disabled={!file || loading}
              sx={{ borderRadius: 2, px: 4, minWidth: 160 }}
            >
              {loading ? (
                <>
                  <CircularProgress size={24} color="inherit" sx={{ mr: 1 }} />
                  Analyzing...
                </>
              ) : (
                "Analyze Data"
              )}
            </Button>
          </Stack>
        </Paper>

        {/* Final Decision Banner */}
        {result && !result.error && result.final_decision && (
          <Paper
            sx={{
              mb: 4,
              p: 4,
              borderRadius: 4,
              background: "linear-gradient(135deg, #1e88e5, #42a5f5)",
              color: "#fff",
              textAlign: "center"
            }}
          >
            <Typography variant="h5" fontWeight="bold" gutterBottom display="flex" justifyContent="center" alignItems="center" gap={1}>
              <WorkspacePremium /> Final AI Decision
            </Typography>
            <Typography variant="h6">Best Model: <b>{result.final_decision.model}</b></Typography>
            <Typography variant="h6">Prediction: <b>{result.final_decision.prediction}</b></Typography>
            <Typography variant="body1">
              Confidence: <b>{result.final_decision.confidence}%</b>
            </Typography>

            <Typography variant="body1">
              Score: <b>{result.final_decision.score}</b>
            </Typography>
          </Paper>
        )}

        {/* Results Section (Grid) */}
        {result && !result.error && (
          <Grid container spacing={4} justifyContent="center" mb={5}>
            {MODEL_CONFIGS.map(({ key, title, color, icon, bg, border }, index) => {
              const data = result[key];
              if (!data) return null;

              const isLastRow = index >= MODEL_CONFIGS.length - 2;

              return (
                <Grid
                  item
                  xs={12}
                  sm={6}
                  md={isLastRow ? 6 : 4}
                  lg={isLastRow ? 6 : 4}
                  key={key}
                >
                  <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column', overflow: "hidden" }}>
                    <Box sx={{ p: 2, display: 'flex', alignItems: 'center', gap: 1.5, borderBottom: `1px solid ${border}`, bgcolor: bg }}>
                      {React.cloneElement(icon, { sx: { color } })}
                      <Typography variant="h6" fontWeight="bold" color={color}>
                        {title}
                      </Typography>
                    </Box>

                    <CardContent sx={{ p: 4, flexGrow: 1 }}>

                      <Stack spacing={1} mb={2}>
                        <Chip
                          label={`Prediction: ${data.prediction || "N/A"}`}
                          sx={{ bgcolor: color, color: "#fff", fontWeight: "bold", width: "fit-content" }}
                        />
                        <Chip
                          label={`Confidence: ${data.confidence || 0}%`}
                          variant="outlined"
                          sx={{ fontWeight: "bold", width: "fit-content", borderColor: color, color }}
                        />
                        <Chip
                          label={`Accuracy: ${data.accuracy || 0}%`}
                          variant="outlined"
                          sx={{ fontWeight: "bold", width: "fit-content", borderColor: color, color }}
                        />
                      </Stack>

                      <Divider sx={{ my: 3 }} />

                      <Grid container spacing={2}>
                        <Grid item xs={6}>
                          <Typography variant="subtitle2" color="textSecondary" textTransform="uppercase" gutterBottom>
                            Brain Waves
                          </Typography>
                          <Typography variant="body2"><b>Alpha:</b> {data.brain_waves?.alpha?.toFixed(3)}</Typography>
                          <Typography variant="body2"><b>Beta:</b> {data.brain_waves?.beta?.toFixed(3)}</Typography>
                          <Typography variant="body2"><b>Gamma:</b> {data.brain_waves?.gamma?.toFixed(3)}</Typography>
                          <Typography variant="body2"><b>Theta:</b> {data.brain_waves?.theta?.toFixed(3)}</Typography>
                        </Grid>

                        <Grid item xs={6}>
                          <Typography variant="subtitle2" color="textSecondary" textTransform="uppercase" gutterBottom>
                            Analysis
                          </Typography>
                          <Typography variant="body2"><b>Condition:</b> {data.analysis?.condition}</Typography>
                          <Typography variant="body2"><b>Cognitive:</b> {data.analysis?.cognitive_activity}</Typography>
                          <Typography variant="body2"><b>Focus:</b> {data.analysis?.focus_score?.toFixed(2)}</Typography>
                          <Typography variant="body2"><b>Stress:</b> {data.analysis?.stress_index?.toFixed(2)}</Typography>
                        </Grid>
                      </Grid>

                      <Box height={220} mt={4}>
                        <ResponsiveContainer>
                          <BarChart data={getChartData(data.brain_waves)} margin={{ top: 10, right: 10, left: -25, bottom: 0 }}>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#eee" />
                            <XAxis dataKey="name" tick={{ fill: '#666', fontSize: 12 }} axisLine={false} tickLine={false} />
                            <YAxis tick={{ fill: '#666', fontSize: 12 }} axisLine={false} tickLine={false} />
                            <Tooltip cursor={{ fill: '#f5f5f5' }} contentStyle={{ borderRadius: 8, border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }} />
                            <Bar dataKey="value" fill={color} radius={[6, 6, 0, 0]} />
                          </BarChart>
                        </ResponsiveContainer>
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
              );
            })}
          </Grid>
        )}

        {/* Global Model Comparison */}
        {result && !result.error && (
          <Paper sx={{ mb: 5, p: 4, borderRadius: 4 }}>
            <Typography variant="h6" fontWeight="bold" gutterBottom>
              📊 Model Performance Comparison
            </Typography>
            <Box sx={{ width: "100%", height: 400 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={comparisonData} margin={{ top: 20, right: 20, left: 0, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#eee" />

                  <XAxis dataKey="name" />
                  <YAxis />

                  <Tooltip
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        const data = payload[0].payload;

                        return (
                          <Paper sx={{ p: 2, borderRadius: 2 }}>
                            <Typography fontWeight="bold">{data.name}</Typography>
                            <Typography variant="body2">Prediction: {data.prediction}</Typography>
                            <Typography variant="body2">Confidence: {data.confidence}%</Typography>
                            <Typography variant="body2">Accuracy: {data.accuracy}%</Typography>
                            <Typography variant="body2">Score: {data.score.toFixed(2)}</Typography>
                          </Paper>
                        );
                      }
                      return null;
                    }}
                  />

                  {/* 🔵 Confidence */}
                  <Bar dataKey="confidence" fill="#42a5f5" radius={[4, 4, 0, 0]} />

                  {/* 🟢 Accuracy */}
                  <Bar dataKey="accuracy" fill="#66bb6a" radius={[4, 4, 0, 0]} />

                  {/* 🟣 Score */}
                  <Bar dataKey="score" fill="#ab47bc" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Box>
          </Paper>
        )}

        {/* Error Message */}
        {result?.error && (
          <Paper sx={{ mt: 4, p: 3, bgcolor: '#ffebee', color: '#c62828', borderRadius: 2, display: 'flex', alignItems: 'center', gap: 2 }}>
            <ErrorOutlineOutlined />
            <Box>
              <Typography variant="subtitle1" fontWeight="bold">Error analyzing data:</Typography>
              <Typography>{result.error}</Typography>
            </Box>
          </Paper>
        )}

        {/* Footer Info (Restored completely) */}
        {result && !result.error && (
          <Paper sx={{ mt: 6, p: 4, borderRadius: 3, bgcolor: "#f9fafb" }}>
            <Typography variant="h5" fontWeight="bold" gutterBottom>
              📊 Understanding the Results
            </Typography>

            <Typography variant="body2" color="text.secondary" paragraph>
              <b>Prediction:</b> Indicates the detected mental state such as stress, relaxation, or normal condition based on EEG patterns.
            </Typography>

            <Typography variant="body2" color="text.secondary" paragraph>
              <b>Confidence:</b> Shows how certain the model is about its prediction. Higher confidence means stronger reliability of the result.
            </Typography>

            <Typography variant="body2" color="text.secondary" paragraph>
              <b>Accuracy:</b> Indicates the overall performance of the model on the training dataset. A higher accuracy means the model has learned patterns effectively, but real-time predictions may still vary depending on input data.
            </Typography>

            <Typography variant="body2" color="text.secondary" paragraph>
              <b>Brain Waves:</b> Alpha (relaxation), Beta (active thinking/stress), Gamma (high focus), Theta (deep relaxation or drowsiness). These signals represent brain activity patterns.
            </Typography>

            <Typography variant="body2" color="text.secondary" paragraph>
              <b>Condition:</b> Overall mental state derived from brain waves, indicating whether the user is relaxed or stressed.
            </Typography>

            <Typography variant="body2" color="text.secondary" paragraph>
              <b>Cognitive Activity:</b> Reflects the level of mental processing and attention based on gamma wave activity.
            </Typography>

            <Typography variant="body2" color="text.secondary" paragraph>
              <b>Focus Score:</b> Represents concentration level. Higher values indicate better focus, while lower values suggest distraction.
            </Typography>

            <Typography variant="body2" color="text.secondary" paragraph>
              <b>Score:</b> A weighted metric combining model confidence and accuracy (0.6 × confidence + 0.4 × accuracy). This helps select the most reliable model by balancing prediction certainty with overall model performance.
            </Typography>

            <Typography variant="body2" color="text.secondary">
              <b>Stress Index:</b> Measures mental stress level using brain signal ratios. Higher values indicate higher stress levels.
            </Typography>


          </Paper>
        )}
      </Container>
    </ThemeProvider>
  );
}