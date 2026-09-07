import {createRequire} from 'node:module';
import test from 'node:test';
import assert from 'node:assert/strict';
const require=createRequire(new URL('../frontend/package.json',import.meta.url));
const {JSDOM}=require('jsdom');
const {parseDraft,parseSettings}=require('../extension/parser.js');
const doc=html=>new JSDOM(html).window.document;
test('reads explicit ESPN IDs and mapped teams, sorts and deduplicates',()=>{
 const row='<div data-pick-number="1" data-team-name="Alpha"><a href="https://www.espn.com/nfl/player/_/id/123/test">Name</a></div>';
 const r=parseDraft(doc(row+row),{teamMap:{Alpha:0}});assert.equal(r.compatible,true);assert.deepEqual(r.picks,[{number:1,team:0,espn_id:'123'}]);
});
test('rejects name-only players and unknown teams',()=>{assert.equal(parseDraft(doc('<div data-pick-number="1">Some Name</div>')).compatible,false);});
test('rejects partial virtualized history',()=>{const r=parseDraft(doc('<div data-pick-number="4" data-player-id="123" data-team-name="Alpha"></div>'),{teamMap:{Alpha:0}});assert.equal(r.compatible,false);assert.ok(r.errors[0].includes('Missing'));});
test('no recognizable rows fails visibly',()=>{assert.equal(parseDraft(doc('<div>Draft room</div>')).compatible,false);});
test('conflicting duplicate pick fails',()=>{const r=parseDraft(doc('<div data-pick-number="1" data-player-id="123" data-team-name="A"></div><div data-pick-number="1" data-player-id="456" data-team-name="A"></div>'),{teamMap:{A:0}});assert.equal(r.compatible,false);});
test('settings capture keeps evidence without inferring scoring',()=>{assert.deepEqual(parseSettings(doc('<table><tr><td>Reception</td><td>0.5</td></tr></table>')),[['Reception','0.5']]);});
