// "use client";
// import { useEffect, useRef, useState } from "react";

// const Button = ({ onClick, children }: any) => (
//   <button onClick={onClick} className="px-6 py-3 bg-white text-black font-bold rounded hover:bg-gray-200 z-50 pointer-events-auto shadow-lg transition-all">
//     {children}
//   </button>
// );

// export default function VideoPlayer() {
//   const [imageSrc, setImageSrc] = useState<string | null>(null);
//   const [isConnected, setIsConnected] = useState(false);
//   const [score, setScore] = useState("Waiting for Game...");
  
//   const socketRef = useRef<WebSocket | null>(null);
//   const captureVideoRef = useRef<HTMLVideoElement>(null); 
//   const captureCanvasRef = useRef<HTMLCanvasElement>(null); 
  
//   const startBroadcast = async () => {
//     try {
//       const stream = await navigator.mediaDevices.getDisplayMedia({
//         video: { width: 1280, height: 720, frameRate: 30 },
//         audio: false,
//       });

//       if (captureVideoRef.current) {
//         captureVideoRef.current.srcObject = stream;
//         captureVideoRef.current.play();
//         connectWebSocket();
//       }
//     } catch (err) {
//       console.error("Error sharing:", err);
//     }
//   };

//   const connectWebSocket = () => {
//     if (socketRef.current) socketRef.current.close();
    
//     const ws = new WebSocket("ws://localhost:8000/ws/stream");
//     socketRef.current = ws;

//     ws.onopen = () => {
//       setIsConnected(true);
//       startSendingFrames();
//     };

//     ws.onmessage = (event) => {
//       const payload = JSON.parse(event.data);
//       if (payload.type === "video_frame") setImageSrc(`data:image/jpeg;base64,${payload.data}`);
//       if (payload.type === "score_update") setScore(payload.data);
//     };

//     ws.onclose = () => setIsConnected(false);
//   };

//   const startSendingFrames = () => {
//     const sendFrame = () => {
//       if (!captureVideoRef.current || !captureCanvasRef.current || !socketRef.current) return;
//       if (socketRef.current.readyState !== WebSocket.OPEN) return;

//       const ctx = captureCanvasRef.current.getContext("2d");
//       if (ctx) {
//         ctx.drawImage(captureVideoRef.current, 0, 0, 1280, 720);
//         const dataURL = captureCanvasRef.current.toDataURL("image/jpeg", 0.6);
//         socketRef.current.send(dataURL);
//       }
//       requestAnimationFrame(sendFrame);
//     };
//     sendFrame();
//   };

//   return (
//     <div className="relative w-full aspect-video bg-black rounded-xl overflow-hidden shadow-2xl border border-gray-800">
//       <video ref={captureVideoRef} className="absolute opacity-0 pointer-events-none" muted playsInline />
//       <canvas ref={captureCanvasRef} width={1280} height={720} className="hidden" />

//       {imageSrc ? (
//         <img src={imageSrc} alt="Live Broadcast" className="w-full h-full object-contain" />
//       ) : (
//         <div className="flex flex-col items-center justify-center h-full space-y-4">
//           <div className="text-5xl animate-bounce">🏀</div>
//           <Button onClick={startBroadcast}>Start Analysis Broadcast</Button>
//           <p className="text-gray-500 text-sm">Select your game tab to begin</p>
//         </div>
//       )}

//       {isConnected && (
//         <div className="absolute top-4 left-4 flex items-center space-x-3">
//             <span className="px-2 py-1 bg-red-600 text-white text-xs font-bold rounded animate-pulse">LIVE</span>
//             <div className="bg-black/70 px-3 py-1 rounded border border-white/20 text-xs font-mono text-yellow-400">
//                 ACTION: {score}
//             </div>
//         </div>
//       )}
//     </div>
//   );
// }

"use client";
import { useEffect, useRef, useState } from "react";

const Button = ({ onClick, children }: any) => (
  <button onClick={onClick} className="px-6 py-3 bg-white text-black font-bold rounded hover:bg-gray-200 z-50 pointer-events-auto shadow-lg transition-all">
    {children}
  </button>
);

export default function VideoPlayer() {
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [score, setScore] = useState("Waiting for Game...");
  const [fps, setFps] = useState(0); // Add FPS counter
  
  const socketRef = useRef<WebSocket | null>(null);
  const captureVideoRef = useRef<HTMLVideoElement>(null); 
  const captureCanvasRef = useRef<HTMLCanvasElement>(null); 
  const frameCountRef = useRef(0);
  const lastTimeRef = useRef(Date.now());

  const startBroadcast = async () => {
    try {
      const stream = await navigator.mediaDevices.getDisplayMedia({
        video: { width: 1280, height: 720, frameRate: 60 }, // Capture High Res
        audio: false,
      });

      if (captureVideoRef.current) {
        captureVideoRef.current.srcObject = stream;
        captureVideoRef.current.play();
        connectWebSocket();
      }
    } catch (err) {
      console.error("Error sharing:", err);
    }
  };

  const connectWebSocket = () => {
    if (socketRef.current) socketRef.current.close();
    
    const ws = new WebSocket("ws://localhost:8000/ws/stream");
    socketRef.current = ws;

    ws.onopen = () => {
      setIsConnected(true);
      startSendingFrames();
    };

    ws.onmessage = (event) => {
      const payload = JSON.parse(event.data);
      if (payload.type === "video_frame") {
        setImageSrc(`data:image/jpeg;base64,${payload.data}`);
        
        // FPS Counter
        frameCountRef.current++;
        const now = Date.now();
        if (now - lastTimeRef.current >= 1000) {
          setFps(frameCountRef.current);
          frameCountRef.current = 0;
          lastTimeRef.current = now;
        }
      }
      if (payload.type === "score_update") setScore(payload.data);
    };

    ws.onclose = () => setIsConnected(false);
  };

  const startSendingFrames = () => {
    const sendFrame = () => {
      if (!captureVideoRef.current || !captureCanvasRef.current || !socketRef.current) return;
      if (socketRef.current.readyState !== WebSocket.OPEN) return;

      const ctx = captureCanvasRef.current.getContext("2d");
      if (ctx) {
        // PERFORMANCE HACK: Draw Small (480p), Display Big
        // This reduces AI load by 4x without losing much accuracy
        ctx.drawImage(captureVideoRef.current, 0, 0, 640, 360);
        
        // Compress to JPEG 0.5 (Fastest)
        const dataURL = captureCanvasRef.current.toDataURL("image/jpeg", 0.5);
        socketRef.current.send(dataURL);
      }
      // Limit to 30 FPS to prevent network clogging
      setTimeout(() => requestAnimationFrame(sendFrame), 1000 / 30); 
    };
    sendFrame();
  };

  return (
    <div className="relative w-full aspect-video bg-black rounded-xl overflow-hidden shadow-2xl border border-gray-800">
      
      {/* Hidden Captures */}
      <video ref={captureVideoRef} className="absolute opacity-0 pointer-events-none" muted playsInline />
      <canvas ref={captureCanvasRef} width={640} height={360} className="hidden" />

      {imageSrc ? (
        <img 
          src={imageSrc} 
          alt="Live Broadcast" 
          className="w-full h-full object-contain"
          style={{ imageRendering: "pixelated" }} // Sharpens the upscale
        />
      ) : (
        <div className="flex flex-col items-center justify-center h-full space-y-4">
          <div className="text-5xl animate-bounce">🏀</div>
          <Button onClick={startBroadcast}>
            Start Analysis Broadcast
          </Button>
          <p className="text-gray-500 text-sm">Select your game tab to begin</p>
        </div>
      )}

      {isConnected && (
        <div className="absolute top-4 left-4 flex items-center space-x-3">
            <span className="px-2 py-1 bg-red-600 text-white text-xs font-bold rounded animate-pulse">
                LIVE
            </span>
            <div className="bg-black/70 px-3 py-1 rounded border border-white/20 text-xs font-mono text-green-400">
                {fps} FPS
            </div>
            <div className="bg-black/70 px-3 py-1 rounded border border-white/20 text-xs font-mono text-yellow-400">
                {score}
            </div>
        </div>
      )}
    </div>
  );
}