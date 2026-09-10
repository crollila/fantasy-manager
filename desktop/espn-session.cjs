'use strict';
const fs = require('node:fs/promises');
const path = require('node:path');

const AUTH_NAMES = new Set(['espn_s2', 'SWID']);
const MAX_AGE = 30 * 24 * 60 * 60 * 1000;
const cookieKey = c => `${c.domain}|${c.path || '/'}|${c.name}`;
function authCookie(c) {
  const host = String(c.domain || '').replace(/^\./, '').toLowerCase();
  return AUTH_NAMES.has(c.name) && (host === 'espn.com' || host.endsWith('.espn.com'));
}
function restorable(c, now = Date.now()) {
  return authCookie(c) && !!c.value && (!c.expirationDate || c.expirationDate * 1000 > now);
}
function cookieDetails(c) {
  const host = c.domain.replace(/^\./, '');
  const result = {url: `https://${host}${c.path || '/'}`, name:c.name, value:c.value, path:c.path || '/', secure:!!c.secure, httpOnly:!!c.httpOnly, sameSite:c.sameSite || 'unspecified'};
  if (!c.hostOnly) result.domain = c.domain;
  if (c.expirationDate) result.expirationDate = c.expirationDate;
  return result;
}

// Only the app's own ESPN session is accessed. Secrets never cross preload or the local API.
// Windows safeStorage uses DPAPI; there is deliberately no plaintext fallback.
class EspnSessionStore {
  constructor(session, safeStorage, profileDir) {
    this.session=session; this.crypto=safeStorage; this.file=path.join(profileDir,'espn-session.enc');
    this.pending=Promise.resolve(); this.problem=null;
  }
  available() {
    return this.crypto.isEncryptionAvailable() && !(process.platform==='linux' && this.crypto.getSelectedStorageBackend?.()==='basic_text');
  }
  async initialize() {
    if (this.available()) {
      try {
        const saved=JSON.parse(this.crypto.decryptString(await fs.readFile(this.file)));
        if (saved.version===1 && Date.now()-saved.savedAt>=0 && Date.now()-saved.savedAt<MAX_AGE) {
          const current=new Set((await this.session.cookies.get({domain:'espn.com'})).filter(c=>restorable(c)).map(cookieKey));
          for (const cookie of saved.cookies || []) {
            if (restorable(cookie) && !current.has(cookieKey(cookie))) await this.session.cookies.set(cookieDetails(cookie));
          }
        }
      } catch(error) { if(error.code!=='ENOENT') this.problem='Saved ESPN session could not be restored. Sign in once to replace it.'; }
    } else this.problem='Windows secure storage is unavailable. ESPN may ask you to sign in after restarting.';
    this.listener=(_event,cookie,_cause,_removed)=>{if(authCookie(cookie))void this.save();};
    this.session.cookies.on('changed',this.listener);
    await this.session.cookies.flushStore();
  }
  save() {
    this.pending=this.pending.then(async()=>{
      if(!this.available())return;
      const cookies=(await this.session.cookies.get({domain:'espn.com'})).filter(c=>restorable(c));
      if(!cookies.length){await fs.rm(this.file,{force:true});return;}
      const encrypted=this.crypto.encryptString(JSON.stringify({version:1,savedAt:Date.now(),cookies}));
      await fs.mkdir(path.dirname(this.file),{recursive:true});
      await fs.writeFile(this.file+'.tmp',encrypted,{mode:0o600});
      await fs.rename(this.file+'.tmp',this.file);
      this.problem=null;
    }).catch(()=>{this.problem='ESPN works for this session, but saving sign-in failed. Check local storage access.';});
    return this.pending;
  }
  async flush() { await this.save(); await this.session.cookies.flushStore(); this.session.flushStorageData(); }
  async status() {
    const cookies=(await this.session.cookies.get({domain:'espn.com'})).filter(c=>restorable(c));
    return {hasSession:cookies.some(c=>c.name==='espn_s2'),remembered:!!this.available(),warning:this.problem};
  }
}
module.exports={EspnSessionStore,authCookie,restorable,cookieDetails};
