'use strict';
function teamLink(value) {
  try {
    const url = new URL(value);
    if (url.origin !== 'https://fantasy.espn.com' || url.pathname !== '/football/team') return null;
    const league_id = url.searchParams.get('leagueId'), team = url.searchParams.get('teamId');
    const season = Number(url.searchParams.get('seasonId') || new Date().getFullYear());
    if (!/^\d+$/.test(league_id || '') || !/^\d+$/.test(team || '') || season < 2005 || season > 2100) return null;
    return {league_id, my_team_id: Number(team), season};
  } catch { return null; }
}
function validOptions(options) {
  const week = Number(options?.week || 1);
  if (!Number.isInteger(week) || week < 1 || week > 18) throw new Error('Choose an NFL week from 1 to 18.');
  const team = options?.url ? teamLink(options.url) : null;
  if (options?.url && !team) throw new Error('Paste the ESPN My Team URL.');
  return {team, week};
}
module.exports = {teamLink, validOptions};
