import React, { useState, useEffect, useRef } from "react";

export default function App() {
  const [isRecording, setIsRecording] = useState(false);
  const [authResult, setAuthResult] = useState("");
  const [wordRecResult, setWordRecResult] = useState("");
  const intervalRef = useRef(null);

  const startRecording = async () => {
    try {
      //alert("Kaam to kr rha");
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
      //setAuthResult("");
      setWordRecResult("");
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
          //alert("call hua bhai");
          const data = await res.json();
            //alert("data mila bhai");
            //setAuthResult(data.authResult);
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
    <div style={{ padding: 20 }}>
      <h1>Recording App</h1>
      {!isRecording ? (
        <button onClick={startRecording}>Start Recording</button>
      ) : (
        <button onClick={stopRecording}>Stop Recording</button>
      )}

      <div style={{ marginTop: 20 }}>
        <h2>Auth Result:</h2>
        {/* <p>{authResult}</p> */}
      </div>

      <div style={{ marginTop: 20 }}>
        <h2>Word Recognition Result:</h2>
        <p>{wordRecResult}</p>
      </div>
    </div>
  );
}
