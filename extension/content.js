let lastSignature='',pending=false;
async function observe(){
 if(pending||!location.pathname.toLowerCase().includes('draft'))return;pending=true;
 try{
  const {config}=await chrome.storage.local.get('config');
  if(!config?.enabled)return;
  const actualLeague=new URL(location.href).searchParams.get('leagueId');
  if(!actualLeague||actualLeague!==config.leagueId){await chrome.runtime.sendMessage({type:'diagnostic',message:'Open the ESPN draft page for the paired league ID.'});return;}
  const parsed=FantasyParser.parseDraft(document,config);
  const signature=JSON.stringify(parsed.picks);
  if(!parsed.compatible){await chrome.runtime.sendMessage({type:'diagnostic',message:parsed.errors.join('\n')||'No recognized draft rows. Inspect selectors in this ESPN draft room before use.'});return;}
  if(signature!==lastSignature){
   const result=await chrome.runtime.sendMessage({type:'snapshot',payload:parsed});
   if(result?.ok)lastSignature=signature;
  }
 }catch(e){console.warn('Fantasy Manager monitor:',e.message);}finally{pending=false;}
}
let debounce;
new MutationObserver(()=>{clearTimeout(debounce);debounce=setTimeout(observe,120);}).observe(document.documentElement,{childList:true,subtree:true,characterData:true});
setInterval(observe,2000);observe();
chrome.storage.onChanged.addListener(()=>{lastSignature='';observe();});
chrome.runtime.onMessage.addListener((message,sender,respond)=>{
 if(message.type==='inspect'){chrome.storage.local.get('config').then(({config})=>respond({draft:FantasyParser.parseDraft(document,config||{}),settingsEvidence:FantasyParser.parseSettings(document)}));return true;}
});
