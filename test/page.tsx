"use client";

/**
 * Ref Zero — Main Page
 * Adds 3-tab navigation: Live Referee | Metrics | About
 * Preserves all existing VideoPlayer, Controls, AgentLog components.
 */

import { useState } from "react";
import VideoPlayer from "@/components/VideoPlayer";
import Controls from "@/components/Controls";
import AgentLog from "@/components/AgentLog";
import About from "@/components/About";
import MetricsDashboard from "@/components/MetricsDashboard";

type Tab = "live" | "metrics" | "about";

export default function Home() {
  const [activeTab, setActiveTab] = useState<Tab>("live");

  const tabs: { id: Tab; label: string; icon: string }[] = [
    { id: "live",    label: "Live Referee",       icon: "🏀" },
    { id: "metrics", label: "Training Metrics",   icon: "📈" },
    { id: "about",   label: "About / Docs",       icon: "📖" },
  ];

  return (
    <main className="min-h-screen bg-black text-white">
      {/* ── Top nav bar ─────────────────────────────────────── */}
      <header className="border-b border-gray-800 bg-gray-950 sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 flex items-center gap-6 h-14">
          {/* Logo */}
          <div className="flex items-center gap-2 flex-shrink-0">
            <span className="text-xl">⚖️</span>
            <span className="font-bold text-white text-lg tracking-tight">Ref Zero</span>
            <span className="text-xs text-gray-500 hidden sm:block ml-1">AI Referee</span>
          </div>

          {/* Tab buttons */}
          <nav className="flex gap-1 ml-4">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                  activeTab === tab.id
                    ? "bg-green-900/50 text-green-400 border border-green-800"
                    : "text-gray-400 hover:text-white hover:bg-gray-800"
                }`}
              >
                <span>{tab.icon}</span>
                <span className="hidden sm:block">{tab.label}</span>
              </button>
            ))}
          </nav>

          {/* Status badge */}
          <div className="ml-auto flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
            <span className="text-xs text-gray-400 hidden sm:block">YOLO11 Active</span>
          </div>
        </div>
      </header>

      {/* ── Tab content ─────────────────────────────────────── */}
      <div className={activeTab === "live" ? "block" : "hidden"}>
        {/*
          Original page layout — kept 100% intact.
          VideoPlayer, Controls, and AgentLog are unchanged.
        */}
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Main video area */}
            <div className="lg:col-span-2 space-y-4">
              <VideoPlayer />
              <Controls />
            </div>
            {/* Agent log sidebar */}
            <div className="lg:col-span-1">
              <AgentLog />
            </div>
          </div>
        </div>
      </div>

      <div className={activeTab === "metrics" ? "block" : "hidden"}>
        <MetricsDashboard />
      </div>

      <div className={activeTab === "about" ? "block" : "hidden"}>
        <About />
      </div>
    </main>
  );
}
