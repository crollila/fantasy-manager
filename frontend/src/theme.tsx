import {useEffect,useState} from 'react';

export type ThemeChoice='system'|'light'|'dark';
const KEY='fm-theme';

function stored():ThemeChoice{try{const v=localStorage.getItem(KEY);return v==='light'||v==='dark'?v:'system';}catch{return 'system';}}
const prefersDark=()=>typeof window!=='undefined'&&!!window.matchMedia?.('(prefers-color-scheme: dark)').matches;
function apply(choice:ThemeChoice){document.documentElement.dataset.theme=choice==='system'?(prefersDark()?'dark':'light'):choice;}

// Applied before the first render so the page never flashes the wrong theme.
apply(stored());

export function useTheme(){
 const [choice,setChoice]=useState<ThemeChoice>(stored);
 useEffect(()=>{apply(choice);try{if(choice==='system')localStorage.removeItem(KEY);else localStorage.setItem(KEY,choice);}catch{/* storage unavailable: theme still applies for this session */}
  if(choice!=='system'||!window.matchMedia)return;const media=window.matchMedia('(prefers-color-scheme: dark)');const listener=()=>apply('system');media.addEventListener?.('change',listener);return()=>media.removeEventListener?.('change',listener);},[choice]);
 return [choice,setChoice] as const;
}

export function ThemeToggle(){
 const [choice,setChoice]=useTheme();
 const next:Record<ThemeChoice,ThemeChoice>={system:'dark',dark:'light',light:'system'};
 const label={system:'Theme: system',dark:'Theme: dark',light:'Theme: light'}[choice];
 return <button className="theme-toggle" onClick={()=>setChoice(next[choice])} title="Switch between system, dark and light themes" aria-label={`${label}. Click to change.`}>{choice==='dark'?'☾':choice==='light'?'☀':'◐'} {label.replace('Theme: ','')}</button>;
}
