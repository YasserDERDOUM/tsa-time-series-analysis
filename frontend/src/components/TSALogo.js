import React from 'react';

export const TSALogo = ({ className = "" }) => {
  return (
    <div className={`flex items-center gap-3 ${className}`}>
      <svg width="50" height="50" viewBox="0 0 50 50" fill="none" xmlns="http://www.w3.org/2000/svg">
        <rect width="50" height="50" rx="10" fill="url(#gradient1)"/>
        <path d="M15 20H20V35H15V20Z" fill="white" opacity="0.9"/>
        <path d="M22 15L27 15L27 35H22V15Z" fill="white" opacity="0.95"/>
        <path d="M29 25C29 25 32 22 35 25C38 28 35 31 35 31V35H29V25Z" fill="white"/>
        <defs>
          <linearGradient id="gradient1" x1="0" y1="0" x2="50" y2="50" gradientUnits="userSpaceOnUse">
            <stop stopColor="#3B82F6"/>
            <stop offset="1" stopColor="#6366F1"/>
          </linearGradient>
        </defs>
      </svg>
      <div>
        <h1 className="text-2xl font-bold text-slate-900 leading-none">TSA</h1>
        <p className="text-xs text-slate-600 font-medium">Time Series Analysis</p>
      </div>
    </div>
  );
};
