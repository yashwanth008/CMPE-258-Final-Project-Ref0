"use client";
import { useState } from "react";

export default function Controls() {
  const [visionEnabled, setVisionEnabled] = useState(true);
  const [isReviewing, setIsReviewing] = useState(false);

  const triggerReview = async () => {
    setIsReviewing(true);
    console.log("🚨 Sending Challenge to Referee...");
    
    try {
      const resp = await fetch("http://localhost:8000/api/trigger_review", {
        method: "POST",
      });
      const data = await resp.json();
      console.log("Server Response:", data);
      
      // Reset button state after 5 seconds
      setTimeout(() => setIsReviewing(false), 5000);
    } catch (e) {
      console.error("Challenge Failed:", e);
      setIsReviewing(false);
    }
  };

  const toggleVision = async () => {
    setVisionEnabled(!visionEnabled);
    try {
      await fetch("http://localhost:8000/api/toggle_vision", { method: "POST" });
    } catch (e) {}
  };

  return (
    <div className="flex items-center space-x-4 bg-black/80 backdrop-blur-md p-2 rounded-full border border-white/10 shadow-2xl pointer-events-auto">
      
      <button
        onClick={toggleVision}
        className={`px-4 py-2 rounded-full text-xs font-bold uppercase tracking-wider transition-all ${
          visionEnabled 
            ? "bg-blue-600/20 text-blue-400 border border-blue-500/50" 
            : "bg-gray-800 text-gray-500 border border-transparent"
        }`}
      >
        {visionEnabled ? "Vision On" : "Vision Off"}
      </button>

      <div className="w-px h-6 bg-white/10" />

      <button
        onClick={triggerReview}
        disabled={isReviewing}
        className={`px-6 py-2 rounded-full font-bold text-sm shadow-lg transition-all transform active:scale-95 flex items-center space-x-2 ${
          isReviewing
            ? "bg-yellow-600 cursor-not-allowed text-white animate-pulse"
            : "bg-red-600 hover:bg-red-500 text-white shadow-red-900/20"
        }`}
      >
        <span>{isReviewing ? "VAR CHECKING..." : "CHALLENGE CALL"}</span>
        {!isReviewing && <span>🚨</span>}
      </button>
    </div>
  );
}



