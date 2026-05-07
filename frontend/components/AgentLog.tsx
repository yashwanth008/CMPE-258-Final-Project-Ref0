"use client";
import { useEffect, useState, useRef } from "react";

type Verdict = {
  id: number;
  timestamp: string;
  sport: string;
  action_breakdown: string;
  verdict: "FOUL" | "CLEAN" | "VIOLATION";
  rule_violated: string;
  explanation: string;
  confidence: number;
};

export default function AgentLog() {
  const [history, setHistory] = useState<Verdict[]>([]);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8000/ws/stream");
    wsRef.current = ws;

    ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        
        if (payload.type === "verdict") {
          let rawText = payload.data;
          // Clean markdown JSON if present
          rawText = rawText.replace(/```json/g, "").replace(/```/g, "").trim();
          
          const data = JSON.parse(rawText);
          
          const newVerdict: Verdict = {
            id: Date.now(),
            timestamp: new Date().toLocaleTimeString(),
            sport: data.sport || "Unknown Sport",
            action_breakdown: data.action_breakdown || "No description provided.",
            verdict: (data.verdict as any) || "UNKNOWN",
            rule_violated: data.rule_violated || "None",
            explanation: data.explanation || "",
            confidence: data.confidence || 0
          };
          
          // Add to TOP of list
          setHistory(prev => [newVerdict, ...prev]);
        }
      } catch (e) {
        // Ignore JSON parse errors from non-verdict messages
      }
    };

    return () => {
      if (ws.readyState === 1) ws.close();
    };
  }, []);

  return (
    <div className="h-full bg-gray-950/50 backdrop-blur-md rounded-xl border border-white/10 overflow-hidden shadow-2xl flex flex-col">
      <div className="p-4 border-b border-white/10 bg-black/40 flex justify-between items-center">
        <h3 className="text-gray-400 text-xs font-bold uppercase tracking-widest">
            Appeal History ({history.length})
        </h3>
        <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
      </div>

      <div className="flex-1 overflow-y-auto custom-scrollbar p-4 space-y-4">
        {history.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full text-gray-600 italic text-sm space-y-2">
                <span>Waiting for appeals...</span>
            </div>
        )}

        {history.map((item) => (
            <div key={item.id} className="bg-gray-900 border border-gray-700 rounded-xl p-4 shadow-lg animate-in fade-in slide-in-from-top-4 duration-500">
                
                {/* Header: Sport & Time */}
                <div className="flex justify-between items-center mb-3">
                    <span className="text-[10px] uppercase tracking-wider text-gray-500 font-bold">{item.sport} • {item.timestamp}</span>
                    <span className={`px-2 py-1 rounded text-[10px] font-bold uppercase tracking-wide border ${
                        item.verdict === 'FOUL' || item.verdict === 'VIOLATION'
                            ? 'bg-red-500/10 text-red-400 border-red-500/20' 
                            : 'bg-green-500/10 text-green-400 border-green-500/20'
                    }`}>
                        {item.verdict}
                    </span>
                </div>

                {/* Analysis Body */}
                <div className="space-y-3">
                    <div>
                        <p className="text-gray-400 text-xs font-bold mb-1">ANALYSIS</p>
                        <p className="text-sm text-gray-200 leading-relaxed">
                            {item.action_breakdown}
                        </p>
                    </div>

                    {item.rule_violated !== "None" && (
                        <div className="bg-red-900/20 border border-red-500/20 p-2 rounded">
                            <p className="text-red-400 text-xs font-bold mb-0.5">⚠️ VIOLATION DETECTED</p>
                            <p className="text-red-300 text-xs font-mono">{item.rule_violated}</p>
                        </div>
                    )}
                    
                    <div className="pt-2 border-t border-gray-800">
                         <p className="text-gray-500 text-xs italic">
                            "{item.explanation}"
                        </p>
                    </div>
                </div>
            </div>
        ))}
      </div>
    </div>
  );
}