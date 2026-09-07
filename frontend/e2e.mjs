import {chromium} from '@playwright/test';
import fs from 'node:fs/promises';
const browser=await chromium.launch({headless:true,channel:'chrome'});
const page=await browser.newPage({viewport:{width:1440,height:1050}});
const errors=[];page.on('pageerror',e=>errors.push(e.message));
try{
 await page.goto('http://127.0.0.1:8000');
 await page.getByRole('heading',{name:'Draft room',exact:true}).waitFor();
 await page.getByText('TOP RECOMMENDATION').waitFor();
 await page.getByRole('button',{name:'Player projections',exact:true}).click();
 await page.getByRole('textbox',{name:'Search projections'}).fill('Demo WR');
 if(await page.locator('tbody tr').count()<1)throw new Error('Projection filtering failed');
 await page.getByRole('button',{name:'Draft board',exact:true}).click();
 if(await page.locator('.team-head').count()!==12)throw new Error('Draft board team count');
 await page.getByRole('button',{name:'League settings',exact:true}).click();
 const json=await page.getByRole('textbox',{name:'League settings JSON'}).inputValue();
 if(JSON.parse(json).id!=='demo')throw new Error('Settings mismatch');
 await page.getByRole('button',{name:'Draft room',exact:true}).click();
 await page.getByRole('textbox',{name:'Search available players'}).fill('');
 await page.screenshot({path:'../storage/ui-desktop.png',fullPage:true});
 await page.setViewportSize({width:390,height:844});
 await page.screenshot({path:'../storage/ui-mobile.png',fullPage:true});
 if(errors.length)throw new Error(errors.join('\n'));
 console.log(JSON.stringify({passed:true,checks:['app loads','projections filter','12-team board','settings roundtrip','desktop screenshot','mobile screenshot'],consoleErrors:errors}));
}finally{await browser.close();}
