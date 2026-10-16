const {contextBridge, ipcRenderer} = require('electron');
contextBridge.exposeInMainWorld('fantasyDesktop', {
  status: () => ipcRenderer.invoke('espn-status'),
  onConnection: callback => {const listener=(_event,state)=>callback(state);ipcRenderer.on('espn-connection-event',listener);return ()=>ipcRenderer.removeListener('espn-connection-event',listener);},
  connect: options => ipcRenderer.invoke('espn-connect', options),
  refresh: options => ipcRenderer.invoke('espn-refresh', options),
  refreshSaved: () => ipcRenderer.invoke('refresh-saved'),
  connections: () => ipcRenderer.invoke('connections'),
  openData: () => ipcRenderer.invoke('open-data'),
  updateStatus: () => ipcRenderer.invoke('update-status'),
  installUpdate: () => ipcRenderer.invoke('update-install'),
  onUpdate: callback => {const listener=(_event,status)=>callback(status);ipcRenderer.on('update-event',listener);return ()=>ipcRenderer.removeListener('update-event',listener);}
});
