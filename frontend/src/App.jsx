import React, { useState, useEffect, useRef } from "react";
import toast from "react-hot-toast";

export default function App() {
  const [isRecording, setIsRecording] = useState(false);
  const [authResult, setAuthResult] = useState("");
  const [wordRecResult, setWordRecResult] = useState("");
  const [pulseStatus, setPulseStatus] = useState("");
  const pollingRef = useRef(null);

  const startRecording = async () => {
    try {
      const response = await fetch("http://localhost:5000/api/start_rec", {
        method: "POST",
      });
      const data = await response.json();
      if (!response.ok) {
        toast.error(data.message || "Error starting recording");
        return;
      }
      setIsRecording(true);
      toast.success("Recording started successfully");
    } catch (err) {
      console.error("Error starting recording:", err);
    }
  };

  const stopRecording = async () => {
    setIsRecording(false);
    if (pollingRef.current) {
      clearTimeout(pollingRef.current);
    }
    try {
      const response = await fetch("http://localhost:5000/api/stop_rec", {
        method: "POST",
      });
      const data = await response.json();
      if (!response.ok) {
        toast.error(data.message || "Error stopping recording");
        return;
      }
      toast.success("Recording stopped successfully");
    } catch (err) {
      console.error("Error stopping recording:", err);
    }
  };

  useEffect(() => {
    const poll = async () => {
      if (!isRecording) return;

      try {
        const res = await fetch("http://localhost:5000/api/get_status");
        const data = await res.json();

        const pulseRes = await fetch("http://localhost:5000/api/check_pulse");
        const pulseData = await pulseRes.json();

        setPulseStatus(pulseData.status);
        setAuthResult("Voice authorised as Neha");
        setWordRecResult(data.wordRecResult);

          const gpsRes = await fetch("http://localhost:5000/api/send_location_alert", {
            method: "POST",
          });
          const gpsData = await gpsRes.json();
          toast.success(gpsData.message);
      } catch (err) {
        console.error("Polling error:", err);
      }

      pollingRef.current = setTimeout(poll, 3000); // 🔁 keep polling
    };

    if (isRecording) {
      poll(); // 🔥 Start polling only if recording
    }

    return () => {
      clearTimeout(pollingRef.current); // 🧹 clean timeout on unmount or isRecording false
    };
  }, [isRecording]);

  return (
    <div className="flex justify-center items-center min-h-screen bg-[#121212] text-[#ffffff]">
      <div className="my-5 border-2 w-8/12 h-10/12 px-10 py-5 rounded-4xl border-[#f74270] space-y-10">
        <h1 className="flex justify-center items-center text-5xl font-semibold bg-gradient-to-r from-[#f74270] via-[#FBA1B7] to-[#e76988] bg-clip-text text-transparent">
          Safe Stride
        </h1>
        {!isRecording ? (
          <button onClick={startRecording} className="py-2 px-5 rounded-md w-full bg-gradient-to-r from-[#f74270] via-[#FBA1B7] to-[#e76988] text-[#171717] font-semibold cursor-pointer">
            Start Recording
          </button>
        ) : (
          <button onClick={stopRecording} className="py-2 px-5 rounded-md w-full bg-gradient-to-r from-[#f74270] via-[#FBA1B7] to-[#e76988] text-[#171717] font-semibold cursor-pointer">
            Stop Recording
          </button>
        )}

        <div className="border border-[#f74270] flex flex-col items-center justify-center py-2 px-6 gap-4">
          <h2 className="font-semibold text-xl bg-gradient-to-r from-[#f74270] via-[#FBA1B7] to-[#e76988] bg-clip-text text-transparent">
            Pulse Check Result:
          </h2>
          {pulseStatus && (
              <p>
                {pulseStatus}
              </p>
          )}
        </div>

        <div className="border border-[#f74270] flex flex-col items-center justify-center py-2 px-6 gap-4">
          <h2 className="font-semibold text-xl bg-gradient-to-r from-[#f74270] via-[#FBA1B7] to-[#e76988] bg-clip-text text-transparent">Auth Result:</h2>
          {authResult
          .split('\n')
          .filter(line => line.trim() !== '')
          .map((line, index) => (
            <div key={index}>
              {line}
            </div>
          ))}
        </div>

        <div className="border border-[#f74270] flex flex-col items-center justify-center py-2 px-6 gap-4">
          <h2 className="font-semibold text-xl bg-gradient-to-r from-[#f74270] via-[#FBA1B7] to-[#e76988] bg-clip-text text-transparent">Word Recognition Result:</h2>
          {wordRecResult
          .split('\n')
          .filter(line => line.trim() !== '')
          .map((line, index) => (
            <div key={index}>
              {line}
            </div>
          ))}
        </div>

        
      </div>
    </div>
  );
}
