import {chromium} from '@playwright/test';
const browser=await chromium.launch({headless:true,channel:'chrome'});
const page=await browser.newPage({viewport:{width:1440,height:1100}});
const errors=[];page.on('pageerror',e=>errors.push(e.message));
const player={id:'fixture-b',name:'Fixture healthy receiver',position:'WR',team:'DEMO',mean:14.2,p10:5.5,p90:25.6,boom:.19,bust:.2,injury_status:'ACTIVE',play_probability:1,locked:false,warnings:['Synthetic UI fixture'],injury:{},game:{opponent:'TEST'},usage:{targets:7.5}};
await page.route('**/api/leagues/*/lineup',route=>route.fulfill({json:{starters:[{slot:'WR',player}],bench:[{...player,id:'fixture-a',name:'Fixture injured receiver',mean:0,p10:0,p90:0,boom:0,bust:1,injury_status:'OUT',play_probability:0}],projected_points:14.2,p10:5.5,p90:25.6,improvement:14.2,start:[player.name],sit:['Fixture injured receiver'],empty_slots:0,roster_updated:'2026-09-07T00:00:00Z',injury_source:{status:'fresh',fetched_at:'2026-09-07T00:00:00Z'},notes:['Synthetic UI fixture; not actual lineup advice.'],missing_projections:[]}}));
try{
 await page.goto('http://127.0.0.1:8000');
 await page.getByRole('button',{name:'My lineup',exact:true}).click();
 await page.getByRole('button',{name:'Analyze my lineup & injuries'}).click();
 await page.getByRole('heading',{name:'14.2 projected points'}).waitFor();
 if(await page.locator('tbody tr').count()!==2)throw new Error('Missing lineup/bench comparison');
 await page.getByRole('button',{name:'Fixture healthy receiver',exact:true}).click();
 await page.getByRole('heading',{name:'Recent usage'}).waitFor();
 await page.getByRole('button',{name:'Close',exact:true}).click();
 await page.screenshot({path:'../storage/lineup-ui.png',fullPage:true});
 if(errors.length)throw new Error(errors.join('\n'));
 console.log('Weekly lineup rendering, injury exclusions, comparisons and detail modal passed.');
}finally{await browser.close();}
