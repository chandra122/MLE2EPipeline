import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  Container, Typography, Box, TextField, Button, Select, MenuItem, FormControl, InputLabel,
  Grid, Paper, CircularProgress, Snackbar, Alert, AppBar, Toolbar
} from '@mui/material';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
});

function App() {
  const [prediction, setPrediction] = useState(null);
  const [responseTime, setResponseTime] = useState(null);
  const [graph, setGraph] = useState(null);
  const [modelType, setModelType] = useState('linear');
  const [modelPerformance, setModelPerformance] = useState(null);
  const [featureImportance, setFeatureImportance] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchModelPerformance();
  }, []);

  const fetchModelPerformance = async () => {
    try {
      setLoading(true);
      const response = await axios.get('http://127.0.0.1:5000/model_performance');
      setModelPerformance(response.data);
    } catch (error) {
      console.error('Error fetching model performance:', error);
      setError('Failed to fetch model performance');
    } finally {
      setLoading(false);
    }
  };

  const handlePredict = async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    const data = Object.fromEntries(formData.entries());

    try {
      setLoading(true);
      const response = await axios.post('http://127.0.0.1:5000/predict', { data, model: modelType });
      setPrediction(response.data.prediction);
      setResponseTime(response.data.response_time);
      setError(null);
    } catch (error) {
      console.error('Error making prediction:', error);
      setError('Failed to make prediction');
    } finally {
      setLoading(false);
    }
  };
  const handleGenerateGraph = async () => {
    const formData = new FormData(document.getElementById('predictionForm'));
    const data = Object.fromEntries(formData.entries());
  
    try {
      setLoading(true);
      const response = await axios.post('http://127.0.0.1:5000/monitor', { data });
      const graphData = response.data;
  
      // Transform response_times and predictions into the format expected by LineChart
      const responseTimesData = graphData.response_times.map((time, index) => ({
        iteration: index + 1,
        time,
      }));
  
      const predictionsData = graphData.predictions.map((prediction, index) => ({
        iteration: index + 1,
        prediction,
      }));
  
      setGraph({
        responseTimesData,
        predictionsData,
        avg_response_time: graphData.avg_response_time,
        avg_prediction: graphData.avg_prediction,
      });
    } catch (error) {
      console.error('Error generating graph:', error);
      setError('Failed to generate graph');
    } finally {
      setLoading(false);
    }
  };

  const handleFeatureImportance = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`http://127.0.0.1:5000/feature_importance?model=${modelType}`, { responseType: 'arraybuffer' });
      const base64 = btoa(
        new Uint8Array(response.data).reduce((data, byte) => data + String.fromCharCode(byte), '')
      );
      setFeatureImportance(`data:image/png;base64,${base64}`);
    } catch (error) {
      console.error('Error fetching feature importance:', error);
      setError('Failed to fetch feature importance');
    } finally {
      setLoading(false);
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6">CO2 Emission Prediction</Typography>
        </Toolbar>
      </AppBar>
      <Container maxWidth="lg">
        <Box my={4}>
          <Paper elevation={3}>
            <Box p={3}>
              <Typography variant="h4" gutterBottom>Prediction Form</Typography>
              <form id="predictionForm" onSubmit={handlePredict}>
                <Grid container spacing={3}>
                  <Grid item xs={12} sm={6}>
                    <FormControl fullWidth>
                      <InputLabel>Model</InputLabel>
                      <Select
                        value={modelType}
                        onChange={(e) => setModelType(e.target.value)}
                        label="Model"
                      >
                        <MenuItem value="linear">Linear Regression</MenuItem>
                        <MenuItem value="decision_tree">Decision Tree</MenuItem>
                        <MenuItem value="random_forest">Random Forest</MenuItem>
                      </Select>
                    </FormControl>
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <TextField fullWidth label="Make" name="Make" required />
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <TextField fullWidth label="Model Year" name="Model_Year" type="number" required />
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <TextField fullWidth label="Engine Size" name="Engine_Size" type="number" step="0.1" required />
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <TextField fullWidth label="Cylinders" name="Cylinders" type="number" required />
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <TextField fullWidth label="Transmission" name="Transmission" required />
                  </Grid>
                  <Grid item xs={12} sm={4}>
                    <TextField fullWidth label="Fuel Consumption City" name="Fuel_Consumption_in_City(L/100 km)" type="number" step="0.1" required />
                  </Grid>
                  <Grid item xs={12} sm={4}>
                    <TextField fullWidth label="Fuel Consumption Hwy" name="Fuel_Consumption_in_City_Hwy(L/100 km)" type="number" step="0.1" required />
                  </Grid>
                  <Grid item xs={12} sm={4}>
                    <TextField fullWidth label="Fuel Consumption Comb" name="Fuel_Consumption_comb(L/100km)" type="number" step="0.1" required />
                  </Grid>
                  <Grid item xs={12}>
                    <TextField fullWidth label="Smog Level" name="Smog_Level" type="number" required />
                  </Grid>
                </Grid>
                <Box mt={3}>
                  <Button type="submit" variant="contained" color="primary" disabled={loading}>
                    {loading ? <CircularProgress size={24} /> : 'Predict'}
                  </Button>
                </Box>
              </form>
            </Box>
          </Paper>

          {prediction && (
            <Paper elevation={3}>
              <Box p={3} mt={3}>
                <Typography variant="h5" gutterBottom>Prediction Results</Typography>
                <Typography>Predicted CO2 Emissions: {prediction.toFixed(2)}</Typography>
                <Typography>Response Time: {responseTime.toFixed(4)} seconds</Typography>
              </Box>
            </Paper>
          )}

          <Box mt={3}>
            <Button onClick={handleGenerateGraph} variant="contained" color="secondary" disabled={loading} style={{marginRight: '10px'}}>
              Generate Response Time Graph
            </Button>
            <Button onClick={handleFeatureImportance} variant="contained" color="primary" disabled={loading}>
              Show Feature Importance
            </Button>
          </Box>

          {graph && (
            <Paper elevation={3}>
             <Box p={3} mt={3}>
              <Typography variant="h5" gutterBottom>Performance Graphs</Typography>
              <Typography variant="h6">Response Time vs Iteration</Typography>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={graph.responseTimesData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="iteration" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="time" stroke="#8884d8" activeDot={{ r: 8 }} />
                </LineChart>
              </ResponsiveContainer>
              <Typography variant="h6">Prediction vs Iteration</Typography>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={graph.predictionsData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="iteration" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="prediction" stroke="#82ca9d" activeDot={{ r: 8 }} />
                </LineChart>
              </ResponsiveContainer>
              <Typography>Average Response Time: {graph.avg_response_time.toFixed(4)} seconds</Typography>
              <Typography>Average Prediction: {graph.avg_prediction.toFixed(2)}</Typography>
            </Box>
          </Paper>
        )}
          {featureImportance && (
            <Paper elevation={3}>
              <Box p={3} mt={3}>
                <Typography variant="h5" gutterBottom>Feature Importance</Typography>
                <img src={featureImportance} alt="Feature Importance Graph" style={{width: '100%'}} />
              </Box>
            </Paper>
          )}

          {modelPerformance && (
            <Paper elevation={3}>
              <Box p={3} mt={3}>
                <Typography variant="h5" gutterBottom>Model Performance</Typography>
                {Object.entries(modelPerformance).map(([model, performance]) => (
                  <Box key={model} mb={2}>
                    <Typography variant="h6">{model.charAt(0).toUpperCase() + model.slice(1)}</Typography>
                    <Typography>MSE: {performance.MSE.toFixed(4)}</Typography>
                    <Typography>R2 Score: {performance.R2.toFixed(4)}</Typography>
                    <Typography>Cross-Validation Mean Score: {performance.CV_mean.toFixed(4)}</Typography>
                  </Box>
                ))}
              </Box>
            </Paper>
          )}
        </Box>
      </Container>
      <Snackbar open={!!error} autoHideDuration={6000} onClose={() => setError(null)}>
        <Alert onClose={() => setError(null)} severity="error">
          {error}
        </Alert>
      </Snackbar>
    </ThemeProvider>
  );
}

export default App;
