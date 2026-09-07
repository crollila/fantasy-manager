const {contextBridge, ipcRenderer} = require('electron');
contextBridge.exposeInMainWorld('fantasyDesktop', {
  connect: options => ipcRenderer.invoke('espn-connect', options),
  refresh: options => ipcRenderer.invoke('espn-refresh', options),
  connections: () => ipcRenderer.invoke('connections'),
  openData: () => ipcRenderer.invoke('open-data')
});
