import "@/App.css";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import TimeSeriesAnalysis from "@/pages/TimeSeriesAnalysis";
import { Toaster } from "@/components/ui/sonner";

function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<TimeSeriesAnalysis />} />
        </Routes>
      </BrowserRouter>
      <Toaster position="top-right" richColors />
    </div>
  );
}

export default App;