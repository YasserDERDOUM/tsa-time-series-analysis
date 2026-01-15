import React, { useState } from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Loader2, Upload, BarChart3, TrendingUp, Activity, Sparkles, Copy, CheckCircle2, FileSpreadsheet } from 'lucide-react';
import { toast } from 'sonner';
import { TSALogo } from '@/components/TSALogo';
import { Footer } from '@/components/Footer';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

export default function TimeSeriesAnalysis() {
  // File upload state
  const [file, setFile] = useState(null);
  const [fileInfo, setFileInfo] = useState(null);
  const [uploading, setUploading] = useState(false);

  // Column selection
  const [dateColumn, setDateColumn] = useState('');
  const [valueColumn, setValueColumn] = useState('');
  const [duplicateStrategy, setDuplicateStrategy] = useState('mean');

  // Analysis results
  const [analyzing, setAnalyzing] = useState(false);
  const [analysisResults, setAnalysisResults] = useState(null);

  // Forecast parameters
  const [p, setPLower] = useState(1);
  const [d, setDLower] = useState(1);
  const [q, setQLower] = useState(1);
  const [seasonal, setSeasonal] = useState(false);
  const [P, setPUpper] = useState(0);
  const [D, setDUpper] = useState(0);
  const [Q, setQUpper] = useState(0);
  const [m, setM] = useState(12);
  const [forecastHorizon, setForecastHorizon] = useState(12);
  const [forecasting, setForecasting] = useState(false);
  const [forecastResults, setForecastResults] = useState(null);

  // AI report
  const [generatingReport, setGeneratingReport] = useState(false);
  const [aiReport, setAiReport] = useState('');
  const [reportMode, setReportMode] = useState('court');
  const [copied, setCopied] = useState(false);

  // Handle file upload
  const handleFileUpload = async (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) return;

    setFile(selectedFile);
    setUploading(true);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const response = await axios.post(`${API}/timeseries/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      setFileInfo(response.data);
      setDateColumn('');
      setValueColumn('');
      setAnalysisResults(null);
      setForecastResults(null);
      setAiReport('');
      toast.success('Fichier uploadé avec succès');
    } catch (error) {
      toast.error('Erreur lors de l\'upload: ' + (error.response?.data?.detail || error.message));
    } finally {
      setUploading(false);
    }
  };

  // Analyze time series
  const handleAnalyze = async () => {
    if (!fileInfo || !dateColumn || !valueColumn) {
      toast.error('Veuillez sélectionner les colonnes date et valeur');
      return;
    }

    setAnalyzing(true);
    try {
      const response = await axios.post(`${API}/timeseries/analyze`, {
        file_id: fileInfo.file_id,
        date_column: dateColumn,
        value_column: valueColumn,
        duplicate_strategy: duplicateStrategy
      });

      setAnalysisResults(response.data);
      toast.success('Analyse complétée');
    } catch (error) {
      toast.error('Erreur d\'analyse: ' + (error.response?.data?.detail || error.message));
    } finally {
      setAnalyzing(false);
    }
  };

  // Generate forecast
  const handleForecast = async () => {
    if (!fileInfo || !dateColumn || !valueColumn) {
      toast.error('Veuillez d\'abord analyser les données');
      return;
    }

    setForecasting(true);
    try {
      const response = await axios.post(`${API}/timeseries/forecast`, {
        file_id: fileInfo.file_id,
        date_column: dateColumn,
        value_column: valueColumn,
        duplicate_strategy: duplicateStrategy,
        p: parseInt(p),
        d: parseInt(d),
        q: parseInt(q),
        seasonal: seasonal,
        P: seasonal ? parseInt(P) : 0,
        D: seasonal ? parseInt(D) : 0,
        Q: seasonal ? parseInt(Q) : 0,
        m: seasonal ? parseInt(m) : 12,
        forecast_horizon: parseInt(forecastHorizon)
      });

      if (response.data.success) {
        setForecastResults(response.data);
        toast.success('Prévisions générées');
      } else {
        toast.error('Erreur: ' + response.data.error);
      }
    } catch (error) {
      toast.error('Erreur de prévision: ' + (error.response?.data?.detail || error.message));
    } finally {
      setForecasting(false);
    }
  };

  // Generate AI report
  const handleGenerateReport = async () => {
    if (!analysisResults && !forecastResults) {
      toast.error('Effectuez d\'abord une analyse et des prévisions');
      return;
    }

    setGeneratingReport(true);
    try {
      const analysisData = {
        summary: analysisResults?.summary,
        stl: analysisResults?.stl,
        adf: analysisResults?.adf,
        forecast: forecastResults ? {
          model_info: forecastResults.model_info,
          forecast_summary: {
            mean: forecastResults.forecast_values.reduce((a, b) => a + b, 0) / forecastResults.forecast_values.length,
            min: Math.min(...forecastResults.forecast_values),
            max: Math.max(...forecastResults.forecast_values),
            first: forecastResults.forecast_values[0],
            last: forecastResults.forecast_values[forecastResults.forecast_values.length - 1]
          }
        } : null
      };

      const response = await axios.post(`${API}/timeseries/ai-report`, {
        file_id: fileInfo.file_id,
        analysis_data: analysisData,
        report_mode: reportMode,
        model_type: 'gpt-5-mini'
      });

      setAiReport(response.data.report);
      toast.success('Rapport IA généré');
    } catch (error) {
      toast.error('Erreur de génération: ' + (error.response?.data?.detail || error.message));
    } finally {
      setGeneratingReport(false);
    }
  };

  // Copy report to clipboard
  const handleCopyReport = () => {
    navigator.clipboard.writeText(aiReport);
    setCopied(true);
    toast.success('Rapport copié');
    setTimeout(() => setCopied(false), 2000);
  };



return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 relative overflow-hidden">
      {/* Animated background elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-blue-400/10 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-indigo-400/10 rounded-full blur-3xl animate-pulse delay-700"></div>
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-purple-400/5 rounded-full blur-3xl animate-pulse delay-1000"></div>
      </div>

      <div className="relative z-10 py-8 px-4">
        <div className="max-w-7xl mx-auto">
          {/* Header with Logo */}
          <div className="text-center mb-8 animate-fade-in">
            <div className="flex items-center justify-center mb-4">
              <TSALogo />
            </div>
            <p className="text-slate-600 text-lg">Analyse Professionnelle • ARIMA/SARIMA • Intelligence Artificielle</p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Left Panel - Upload & Parameters */}
            <div className="lg:col-span-1 space-y-6">
              {/* Upload */}
              <Card data-testid="upload-card" className="border-slate-200 shadow-lg hover:shadow-xl transition-shadow duration-300 backdrop-blur-sm bg-white/90">
                <CardHeader className="bg-gradient-to-r from-blue-50 to-indigo-50 border-b border-slate-100">
                  <CardTitle className="flex items-center gap-2 text-blue-900">
                    <div className="p-2 bg-blue-100 rounded-lg">
                      <Upload className="w-5 h-5 text-blue-600" />
                    </div>
                    Upload Fichier
                  </CardTitle>
                  <CardDescription>CSV ou Excel (.xlsx, .xls)</CardDescription>
                </CardHeader>
                <CardContent className="pt-6">
                  <div className="relative group">
                    <Input
                      data-testid="file-input"
                      type="file"
                      accept=".csv,.xlsx,.xls"
                      onChange={handleFileUpload}
                      disabled={uploading}
                      className="cursor-pointer file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-blue-100 file:text-blue-700 hover:file:bg-blue-200 transition-all"
                    />
                  </div>
                  {uploading && (
                    <div className="flex items-center gap-2 mt-3 text-sm text-blue-600 bg-blue-50 p-3 rounded-lg animate-pulse">
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Upload en cours...
                    </div>
                  )}
                  {fileInfo && (
                    <div data-testid="file-info" className="mt-4 p-4 bg-gradient-to-r from-green-50 to-emerald-50 rounded-xl border border-green-200 shadow-sm animate-fade-in">
                      <div className="flex items-start gap-3">
                        <div className="p-2 bg-green-100 rounded-lg">
                          <FileSpreadsheet className="w-5 h-5 text-green-600" />
                        </div>
                        <div>
                          <p className="text-sm font-semibold text-green-900">{fileInfo.filename}</p>
                          <p className="text-xs text-green-700 mt-1">{fileInfo.n_rows} lignes • {fileInfo.columns.length} colonnes</p>
                        </div>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Column Selection */}
              {fileInfo && (
                <Card data-testid="column-selection-card" className="border-slate-200 shadow-lg hover:shadow-xl transition-all duration-300 backdrop-blur-sm bg-white/90 animate-fade-in">
                  <CardHeader className="bg-gradient-to-r from-purple-50 to-pink-50 border-b border-slate-100">
                    <CardTitle className="text-purple-900">Configuration</CardTitle>
                    <CardDescription>Sélectionnez vos colonnes d'analyse</CardDescription>
                  </CardHeader>
                  <CardContent className="pt-6 space-y-4">
                    <div className="space-y-2">
                      <Label className="text-sm font-semibold text-slate-700">Colonne Date/Temps</Label>
                      <Select value={dateColumn} onValueChange={setDateColumn}>
                        <SelectTrigger data-testid="date-column-select" className="hover:border-purple-300 transition-colors">
                          <SelectValue placeholder="Choisir colonne date" />
                        </SelectTrigger>
                        <SelectContent>
                          {fileInfo.columns.map(col => (
                            <SelectItem key={col} value={col}>{col}</SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>

                    <div className="space-y-2">
                      <Label className="text-sm font-semibold text-slate-700">Colonne Valeur</Label>
                      <Select value={valueColumn} onValueChange={setValueColumn}>
                        <SelectTrigger data-testid="value-column-select" className="hover:border-purple-300 transition-colors">
                          <SelectValue placeholder="Choisir colonne valeur" />
                        </SelectTrigger>
                        <SelectContent>
                          {fileInfo.columns.map(col => (
                            <SelectItem key={col} value={col}>{col}</SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>

                    <div className="space-y-2">
                      <Label className="text-sm font-semibold text-slate-700">Stratégie Doublons</Label>
                      <Select value={duplicateStrategy} onValueChange={setDuplicateStrategy}>
                        <SelectTrigger data-testid="duplicate-strategy-select" className="hover:border-purple-300 transition-colors">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="mean">Moyenne</SelectItem>
                          <SelectItem value="sum">Somme</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>

                    <Button 
                      data-testid="analyze-button"
                      onClick={handleAnalyze} 
                      disabled={analyzing || !dateColumn || !valueColumn}
                      className="w-full bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 shadow-lg hover:shadow-xl transition-all duration-300 transform hover:-translate-y-0.5"
                    >
                      {analyzing ? (
                        <>
                          <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                          Analyse en cours...
                        </>
                      ) : (
                        <>
                          <BarChart3 className="w-4 h-4 mr-2" />
                          Lancer l'Analyse
                        </>
                      )}
                    </Button>
                  </CardContent>
                </Card>
              )}

              {/* Forecast Parameters */}
              {analysisResults && (
                <Card data-testid="forecast-card" className="border-slate-200 shadow-lg hover:shadow-xl transition-all duration-300 backdrop-blur-sm bg-white/90 animate-fade-in">
                  <CardHeader className="bg-gradient-to-r from-blue-50 to-cyan-50 border-b border-slate-100">
                    <CardTitle className="text-blue-900">Modèle Prévisionnel</CardTitle>
                    <CardDescription>Configurez ARIMA ou SARIMA</CardDescription>
                  </CardHeader>
                  <CardContent className="pt-6 space-y-4">
                    <div>
                      <Label className="text-sm font-semibold text-slate-700 mb-2 block">Paramètres ARIMA</Label>
                      <div className="grid grid-cols-3 gap-2">
                        <div>
                          <Label className="text-xs text-slate-600">p (AR)</Label>
                          <Input
                            type="number"
                            value={p}
                            onChange={(e) => setPLower(e.target.value)}
                            min="0"
                            className="h-9 hover:border-blue-400 transition-colors"
                          />
                        </div>
                        <div>
                          <Label className="text-xs text-slate-600">d (I)</Label>
                          <Input
                            type="number"
                            value={d}
                            onChange={(e) => setDLower(e.target.value)}
                            min="0"
                            className="h-9 hover:border-blue-400 transition-colors"
                          />
                        </div>
                        <div>
                          <Label className="text-xs text-slate-600">q (MA)</Label>
                          <Input
                            type="number"
                            value={q}
                            onChange={(e) => setQLower(e.target.value)}
                            min="0"
                            className="h-9 hover:border-blue-400 transition-colors"
                          />
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center gap-2 p-3 bg-slate-50 rounded-lg hover:bg-slate-100 transition-colors cursor-pointer" onClick={() => setSeasonal(!seasonal)}>
                      <input
                        type="checkbox"
                        checked={seasonal}
                        onChange={(e) => setSeasonal(e.target.checked)}
                        className="w-4 h-4 accent-blue-600"
                      />
                      <Label className="font-semibold text-slate-700 cursor-pointer">Modèle Saisonnier (SARIMA)</Label>
                    </div>

                    {seasonal && (
                      <div className="animate-fade-in">
                        <Label className="text-sm font-semibold text-slate-700 mb-2 block">Paramètres Saisonniers</Label>
                        <div className="grid grid-cols-4 gap-2">
                          <div>
                            <Label className="text-xs text-slate-600">P</Label>
                            <Input
                              type="number"
                              value={P}
                              onChange={(e) => setPUpper(e.target.value)}
                              min="0"
                              className="h-9 hover:border-cyan-400 transition-colors"
                            />
                          </div>
                          <div>
                            <Label className="text-xs text-slate-600">D</Label>
                            <Input
                              type="number"
                              value={D}
                              onChange={(e) => setDUpper(e.target.value)}
                              min="0"
                              className="h-9 hover:border-cyan-400 transition-colors"
                            />
                          </div>
                          <div>
                            <Label className="text-xs text-slate-600">Q</Label>
                            <Input
                              type="number"
                              value={Q}
                              onChange={(e) => setQUpper(e.target.value)}
                              min="0"
                              className="h-9 hover:border-cyan-400 transition-colors"
                            />
                          </div>
                          <div>
                            <Label className="text-xs text-slate-600">m</Label>
                            <Input
                              type="number"
                              value={m}
                              onChange={(e) => setM(e.target.value)}
                              min="2"
                              className="h-9 hover:border-cyan-400 transition-colors"
                            />
                          </div>
                        </div>
                      </div>
                    )}

                    <div>
                      <Label className="text-sm font-semibold text-slate-700">Horizon de Prévision</Label>
                      <Input
                        type="number"
                        value={forecastHorizon}
                        onChange={(e) => setForecastHorizon(e.target.value)}
                        min="1"
                        className="mt-2 hover:border-blue-400 transition-colors"
                        placeholder="Nombre de périodes"
                      />
                    </div>

                    <Button 
                      data-testid="forecast-button"
                      onClick={handleForecast} 
                      disabled={forecasting}
                      className="w-full bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 shadow-lg hover:shadow-xl transition-all duration-300 transform hover:-translate-y-0.5"
                    >
                      {forecasting ? (
                        <>
                          <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                          Calcul en cours...
                        </>
                      ) : (
                        <>
                          <TrendingUp className="w-4 h-4 mr-2" />
                          Générer Prévisions
                        </>
                      )}
                    </Button>
                  </CardContent>
                </Card>
              )}
            </div>
{/* Right Panel - Visualizations */}
            <div className="lg:col-span-2 space-y-6">
              {analysisResults && (
                <Card data-testid="visualization-card" className="border-slate-200 shadow-xl backdrop-blur-sm bg-white/90 animate-fade-in">
                  <CardHeader className="bg-gradient-to-r from-emerald-50 to-teal-50 border-b border-slate-100">
                    <CardTitle className="flex items-center gap-2 text-emerald-900">
                      <div className="p-2 bg-emerald-100 rounded-lg">
                        <Activity className="w-5 h-5 text-emerald-600" />
                      </div>
                      Visualisations & Analyses
                    </CardTitle>
                    <CardDescription>Explorez vos données en profondeur</CardDescription>
                  </CardHeader>
                  <CardContent className="pt-6">
                    <Tabs defaultValue="original">
                      <TabsList className="grid w-full grid-cols-4 mb-6 bg-slate-100 p-1 rounded-xl">
                        <TabsTrigger value="original" className="data-[state=active]:bg-white data-[state=active]:shadow-md rounded-lg transition-all">
                          Série
                        </TabsTrigger>
                        <TabsTrigger value="decomposition" className="data-[state=active]:bg-white data-[state=active]:shadow-md rounded-lg transition-all">
                          STL
                        </TabsTrigger>
                        <TabsTrigger value="stationarity" className="data-[state=active]:bg-white data-[state=active]:shadow-md rounded-lg transition-all">
                          ADF
                        </TabsTrigger>
                        <TabsTrigger value="forecast" disabled={!forecastResults} className="data-[state=active]:bg-white data-[state=active]:shadow-md rounded-lg transition-all">
                          Prévisions
                        </TabsTrigger>
                      </TabsList>

                      <TabsContent value="original" className="mt-4">
                        <Plot
                          data={[{
                            x: analysisResults.original.dates,
                            y: analysisResults.original.values,
                            type: 'scatter',
                            mode: 'lines',
                            line: { color: '#3b82f6', width: 2 },
                            name: 'Série originale'
                          }]}
                          layout={{
                            title: 'Série Temporelle Originale',
                            xaxis: { title: 'Date' },
                            yaxis: { title: 'Valeur' },
                            height: 400,
                            margin: { t: 50, b: 50, l: 60, r: 40 }
                          }}
                          config={{ responsive: true }}
                          className="w-full"
                        />
                        <div className="mt-4 p-5 bg-gradient-to-br from-slate-50 to-blue-50 rounded-xl border border-slate-200 shadow-sm">
                          <h4 className="font-bold text-slate-800 mb-3 flex items-center gap-2">
                            <BarChart3 className="w-4 h-4 text-blue-600" />
                            Statistiques Descriptives
                          </h4>
                          <div className="grid grid-cols-2 gap-3">
                            <div className="p-3 bg-white rounded-lg shadow-sm">
                              <p className="text-xs text-slate-600">Observations</p>
                              <p className="text-lg font-bold text-blue-600">{analysisResults.summary.n_observations}</p>
                            </div>
                            <div className="p-3 bg-white rounded-lg shadow-sm">
                              <p className="text-xs text-slate-600">Fréquence</p>
                              <p className="text-lg font-bold text-purple-600">{analysisResults.summary.frequency}</p>
                            </div>
                            <div className="p-3 bg-white rounded-lg shadow-sm">
                              <p className="text-xs text-slate-600">Moyenne</p>
                              <p className="text-lg font-bold text-green-600">{analysisResults.summary.mean.toFixed(2)}</p>
                            </div>
                            <div className="p-3 bg-white rounded-lg shadow-sm">
                              <p className="text-xs text-slate-600">Écart-type</p>
                              <p className="text-lg font-bold text-orange-600">{analysisResults.summary.std.toFixed(2)}</p>
                            </div>
                            <div className="p-3 bg-white rounded-lg shadow-sm">
                              <p className="text-xs text-slate-600">Minimum</p>
                              <p className="text-lg font-bold text-red-600">{analysisResults.summary.min.toFixed(2)}</p>
                            </div>
                            <div className="p-3 bg-white rounded-lg shadow-sm">
                              <p className="text-xs text-slate-600">Maximum</p>
                              <p className="text-lg font-bold text-teal-600">{analysisResults.summary.max.toFixed(2)}</p>
                            </div>
                          </div>
                        </div>
                      </TabsContent>

                      <TabsContent value="decomposition" className="mt-4">
                        {analysisResults.stl.success ? (
                          <div className="space-y-4">
                            <Plot
                              data={[{
                                x: analysisResults.stl.dates,
                                y: analysisResults.stl.trend,
                                type: 'scatter',
                                mode: 'lines',
                                line: { color: '#10b981', width: 2 },
                                name: 'Tendance'
                              }]}
                              layout={{
                                title: 'Tendance',
                                xaxis: { title: 'Date' },
                                yaxis: { title: 'Valeur' },
                                height: 250,
                                margin: { t: 40, b: 40, l: 60, r: 40 }
                              }}
                              config={{ responsive: true }}
                              className="w-full"
                            />
                            <Plot
                              data={[{
                                x: analysisResults.stl.dates,
                                y: analysisResults.stl.seasonal,
                                type: 'scatter',
                                mode: 'lines',
                                line: { color: '#f59e0b', width: 2 },
                                name: 'Saisonnalité'
                              }]}
                              layout={{
                                title: 'Saisonnalité',
                                xaxis: { title: 'Date' },
                                yaxis: { title: 'Valeur' },
                                height: 250,
                                margin: { t: 40, b: 40, l: 60, r: 40 }
                              }}
                              config={{ responsive: true }}
                              className="w-full"
                            />
                            <Plot
                              data={[{
                                x: analysisResults.stl.dates,
                                y: analysisResults.stl.resid,
                                type: 'scatter',
                                mode: 'lines',
                                line: { color: '#ef4444', width: 1 },
                                name: 'Résidus'
                              }]}
                              layout={{
                                title: 'Résidus',
                                xaxis: { title: 'Date' },
                                yaxis: { title: 'Valeur' },
                                height: 250,
                                margin: { t: 40, b: 40, l: 60, r: 40 }
                              }}
                              config={{ responsive: true }}
                              className="w-full"
                            />
                          </div>
                        ) : (
                          <Alert>
                            <AlertDescription>{analysisResults.stl.error}</AlertDescription>
                          </Alert>
                        )}
                      </TabsContent>

                      <TabsContent value="stationarity" className="mt-4">
                        <div className="p-6 bg-slate-50 rounded-lg">
                          <h3 className="text-lg font-semibold mb-4">Test de Dickey-Fuller Augmenté (ADF)</h3>
                          {analysisResults.adf.error ? (
                            <Alert>
                              <AlertDescription>{analysisResults.adf.error}</AlertDescription>
                            </Alert>
                          ) : (
                            <div className="space-y-3">
                              <div className="p-4 bg-white rounded-lg border">
                                <p className="text-sm font-medium text-slate-600 mb-1">Interprétation</p>
                                <p className="text-slate-900">{analysisResults.adf.interpretation}</p>
                              </div>
                              <div className="grid grid-cols-2 gap-4">
                                <div className="p-3 bg-white rounded-lg border">
                                  <p className="text-xs font-medium text-slate-600">Statistique ADF</p>
                                  <p className="text-lg font-bold text-slate-900">{analysisResults.adf.adf_statistic.toFixed(4)}</p>
                                </div>
                                <div className="p-3 bg-white rounded-lg border">
                                  <p className="text-xs font-medium text-slate-600">p-value</p>
                                  <p className={`text-lg font-bold ${analysisResults.adf.p_value < 0.05 ? 'text-green-600' : 'text-red-600'}`}>
                                    {analysisResults.adf.p_value.toFixed(4)}
                                  </p>
                                </div>
                              </div>
                              <div className="p-3 bg-white rounded-lg border">
                                <p className="text-sm font-medium text-slate-600 mb-2">Valeurs Critiques</p>
                                <div className="grid grid-cols-3 gap-2 text-sm">
                                  <div>
                                    <p className="text-xs text-slate-600">1%</p>
                                    <p className="font-medium">{analysisResults.adf.critical_values['1%'].toFixed(4)}</p>
                                  </div>
                                  <div>
                                    <p className="text-xs text-slate-600">5%</p>
                                    <p className="font-medium">{analysisResults.adf.critical_values['5%'].toFixed(4)}</p>
                                  </div>
                                  <div>
                                    <p className="text-xs text-slate-600">10%</p>
                                    <p className="font-medium">{analysisResults.adf.critical_values['10%'].toFixed(4)}</p>
                                  </div>
                                </div>
                              </div>
                            </div>
                          )}
                        </div>
                      </TabsContent>

                      <TabsContent value="forecast" className="mt-4">
                        {forecastResults && (
                          <div className="space-y-4">
                            <Plot
                              data={[
                                {
                                  x: forecastResults.in_sample_dates,
                                  y: forecastResults.observed_values,
                                  type: 'scatter',
                                  mode: 'lines',
                                  line: { color: '#3b82f6', width: 2 },
                                  name: 'Observé'
                                },
                                {
                                  x: forecastResults.forecast_dates,
                                  y: forecastResults.forecast_values,
                                  type: 'scatter',
                                  mode: 'lines',
                                  line: { color: '#10b981', width: 2, dash: 'dash' },
                                  name: 'Prévision'
                                },
                                {
                                  x: forecastResults.forecast_dates,
                                  y: forecastResults.upper_ci,
                                  type: 'scatter',
                                  mode: 'lines',
                                  line: { width: 0 },
                                  showlegend: false,
                                  hoverinfo: 'skip'
                                },
                                {
                                  x: forecastResults.forecast_dates,
                                  y: forecastResults.lower_ci,
                                  type: 'scatter',
                                  mode: 'lines',
                                  line: { width: 0 },
                                  fill: 'tonexty',
                                  fillcolor: 'rgba(16, 185, 129, 0.2)',
                                  name: 'IC 95%'
                                }
                              ]}
                              layout={{
                                title: 'Prévisions avec Intervalle de Confiance',
                                xaxis: { title: 'Date' },
                                yaxis: { title: 'Valeur' },
                                height: 400,
                                margin: { t: 50, b: 50, l: 60, r: 40 }
                              }}
                              config={{ responsive: true }}
                              className="w-full"
                            />
                            <div className="p-4 bg-slate-50 rounded-lg">
                              <h4 className="font-semibold mb-2">Informations Modèle</h4>
                              <div className="grid grid-cols-2 gap-2 text-sm">
                                <p><span className="font-medium">Type:</span> {forecastResults.model_info.model_type}</p>
                                <p><span className="font-medium">Ordre:</span> {forecastResults.model_info.order}</p>
                                {forecastResults.model_info.seasonal_order !== "None" && (
                                  <p><span className="font-medium">Ordre saisonnier:</span> {forecastResults.model_info.seasonal_order}</p>
                                )}
                                <p><span className="font-medium">AIC:</span> {forecastResults.model_info.aic.toFixed(2)}</p>
                                <p><span className="font-medium">BIC:</span> {forecastResults.model_info.bic.toFixed(2)}</p>
                              </div>
                            </div>
                          </div>
                        )}
                      </TabsContent>
                    </Tabs>
                  </CardContent>
                </Card>
              )}
{/* AI Report */}
              {analysisResults && (
                <Card data-testid="ai-report-card" className="border-slate-200 shadow-xl backdrop-blur-sm bg-white/90 animate-fade-in">
                  <CardHeader className="bg-gradient-to-r from-purple-50 via-pink-50 to-rose-50 border-b border-slate-100">
                    <CardTitle className="flex items-center gap-2 text-purple-900">
                      <div className="p-2 bg-purple-100 rounded-lg">
                        <Sparkles className="w-5 h-5 text-purple-600" />
                      </div>
                      Rapport Intelligence Artificielle
                    </CardTitle>
                    <CardDescription>Analyse automatique alimentée par GPT-5-mini</CardDescription>
                  </CardHeader>
                  <CardContent className="pt-6 space-y-4">
                    <div className="flex gap-2">
                      <Select value={reportMode} onValueChange={setReportMode}>
                        <SelectTrigger className="w-40 hover:border-purple-400 transition-colors">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="court">Rapport Court</SelectItem>
                          <SelectItem value="long">Rapport Long</SelectItem>
                        </SelectContent>
                      </Select>
                      <Button 
                        data-testid="generate-report-button"
                        onClick={handleGenerateReport} 
                        disabled={generatingReport}
                        className="flex-1 bg-gradient-to-r from-purple-600 via-pink-600 to-rose-600 hover:from-purple-700 hover:via-pink-700 hover:to-rose-700 shadow-lg hover:shadow-xl transition-all duration-300"
                      >
                        {generatingReport ? (
                          <>
                            <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                            Génération IA en cours...
                          </>
                        ) : (
                          <>
                            <Sparkles className="w-4 h-4 mr-2" />
                            Générer Rapport IA
                          </>
                        )}
                      </Button>
                    </div>

                    {aiReport && (
                      <div className="relative animate-fade-in">
                        <div className="p-5 bg-gradient-to-br from-purple-50 via-pink-50 to-rose-50 rounded-xl border border-purple-200 shadow-inner whitespace-pre-wrap text-sm leading-relaxed">
                          {aiReport}
                        </div>
                        <Button
                          data-testid="copy-report-button"
                          onClick={handleCopyReport}
                          size="sm"
                          variant="outline"
                          className="mt-3 border-purple-300 hover:bg-purple-50 transition-colors"
                        >
                          {copied ? (
                            <>
                              <CheckCircle2 className="w-4 h-4 mr-2 text-green-600" />
                              Copié avec succès!
                            </>
                          ) : (
                            <>
                              <Copy className="w-4 h-4 mr-2" />
                              Copier le Rapport
                            </>
                          )}
                        </Button>
                      </div>
                    )}
                  </CardContent>
                </Card>
              )}
            </div>
          </div>
        </div>
      </div>
      
      <Footer />
    </div>
  );
}