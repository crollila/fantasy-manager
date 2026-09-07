const {test}=require('node:test');const assert=require('node:assert/strict');
const {teamLink,validOptions}=require('./connection.cjs');
test('only ESPN My Team links become import targets',()=>{
 assert.deepEqual(teamLink('https://fantasy.espn.com/football/team?leagueId=123&teamId=4&seasonId=2026'),{league_id:'123',my_team_id:4,season:2026});
 for(const url of ['https://evil.test/football/team?leagueId=123&teamId=4','https://fantasy.espn.com.evil.test/football/team?leagueId=123&teamId=4','file:///etc/passwd','https://fantasy.espn.com/football/league?leagueId=123','https://fantasy.espn.com/football/team?leagueId=x&teamId=4']) assert.equal(teamLink(url),null);
});
test('invalid weeks and targets fail before network access',()=>{
 assert.throws(()=>validOptions({week:19}));assert.throws(()=>validOptions({url:'https://evil.test'}));assert.equal(validOptions({week:2}).week,2);
});
