
// Logos are bundled with the app so they render without a network connection.
// Files are named by ESPN abbreviation; map the franchise codes this app emits onto them.
const SLUGS:Record<string,string>={LA:'lar',LAR:'lar',STL:'lar',WAS:'wsh',WSH:'wsh',SD:'lac',OAK:'lv',JAC:'jax',HST:'hou',BLT:'bal',CLV:'cle',ARZ:'ari'};
// Badge colours for the monogram shown when a code has no bundled logo.
const COLORS:Record<string,string>={ARI:'#97233F',ATL:'#A71930',BAL:'#241773',BUF:'#00338D',CAR:'#0085CA',CHI:'#0B162A',CIN:'#FB4F14',CLE:'#311D00',DAL:'#041E42',DEN:'#0A2343',DET:'#0076B6',GB:'#203731',HOU:'#03202F',IND:'#002C5F',JAX:'#006778',KC:'#E31837',LA:'#003594',LAR:'#003594',LAC:'#0080C6',LV:'#101820',MIA:'#008E97',MIN:'#4F2683',NE:'#002244',NO:'#101820',NYG:'#0B2265',NYJ:'#125740',PHI:'#004C54',PIT:'#101820',SEA:'#002244',SF:'#AA0000',TB:'#D50A0A',TEN:'#0C2340',WAS:'#5A1414',WSH:'#5A1414'};

const modules=import.meta.glob('./assets/team-logos/*.png',{eager:true,import:'default'}) as Record<string,string>;
const LOGOS:Record<string,string>=Object.fromEntries(Object.entries(modules).map(([path,url])=>[path.slice(path.lastIndexOf('/')+1,-4),url]));

export function TeamLogo({team,size=30}:{team?:string|null;size?:number}){
  const code=String(team??'').trim().toUpperCase();
  if(!code) return null;
  const src=LOGOS[SLUGS[code]??code.toLowerCase()];
  if(!src) return <span className="team-logo team-logo-text" aria-hidden="true" style={{width:size,height:size,background:COLORS[code]??'#44525e',fontSize:Math.max(8,Math.round(size*.36))}}>{code}</span>;
  return <img className="team-logo" aria-hidden="true" alt="" style={{width:size,height:size}} src={src}/>;
}
