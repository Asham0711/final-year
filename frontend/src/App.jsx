import React, { useState, useEffect, useRef } from "react";

export default function App() {
  const [isRecording, setIsRecording] = useState(false);
  const [authResult, setAuthResult] = useState("");
  const [wordRecResult, setWordRecResult] = useState("");
  const intervalRef = useRef(null);

  const startRecording = async () => {
    try {
      const response = await fetch("http://localhost:5000/api/start_rec", {
        method: "POST",
      });
      const data = await response.json();
      if (!response.ok) {
        alert(data.message || "Error starting recording");
        return;
      }
      setIsRecording(true);
    } catch (err) {
      console.error("Error starting recording:", err);
    }
  };

  const stopRecording = async () => {
    try {
      const response = await fetch("http://localhost:5000/api/stop_rec", {
        method: "POST",
      });
      const data = await response.json();
      if (!response.ok) {
        alert(data.message || "Error stopping recording");
        return;
      }
      setIsRecording(false);
    } catch (err) {
      console.error("Error stopping recording:", err);
    }
  };

  useEffect(() => {
    if (isRecording) {
      // Poll every 3 seconds to get auth.py and wordrecognition.py results
      intervalRef.current = setInterval(async () => {
        try {
          const res = await fetch("http://localhost:5000/api/get_status");
          const data = await res.json();
            setAuthResult(data.authResult);
            setWordRecResult(data.wordRecResult);
            //clearInterval(intervalRef.current);
        } catch (err) {
          console.error("Error fetching status:", err);
        }
      }, 3000);
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    }

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
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
