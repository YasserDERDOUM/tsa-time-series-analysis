import React from 'react';

export const Footer = () => {
  return (
    <footer className="mt-12 py-6 border-t border-slate-200 bg-white/50 backdrop-blur-sm">
      <div className="max-w-7xl mx-auto px-4 text-center">
        <p className="text-sm text-slate-700 font-medium">
          Réalisé par <span className="text-blue-600 font-semibold">Yasser DERDOUM</span> et{' '}
          <span className="text-indigo-600 font-semibold">Aghiles Boudjemaa</span>
        </p>
        <p className="text-xs text-slate-500 mt-1">
          M2-NEXA PARIS • Tous droits réservés © {new Date().getFullYear()}
        </p>
      </div>
    </footer>
  );
};

