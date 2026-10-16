# Fantasy Manager 0.7.1

Adds automatic updates and fixes teams staying on the previous week after a refresh.

## New: automatic updates

This is the last version you need to download and install yourself. From now on the app
checks GitHub Releases when it opens and every six hours, and downloads a new version in the
background. When it is ready a banner offers **Restart to update**. If you ignore it, the
update installs the next time you close the app and is in place when you reopen it. There is
no uninstall step, and saved teams, the ESPN profile and prediction history are kept. The
current version and update progress are shown at the bottom of the home screen. Update
activity is written to `updater.log` in the data folder; an offline check is simply retried
later.

## Fixed: teams stuck on last week

When the app opens it starts a data refresh in the background. The home screen asked for a
refresh as well, found one already running, and treated that as finished straight away. It
then re-synced your ESPN teams using the week from the last *completed* refresh, which could
be a week or more old. Rosters, ESPN projections and lineups stayed on that week (for example
week 5 while week 6 was already under way), even though the NFL games tab moved on a few
minutes later.

- A refresh request now waits for a refresh already in progress and only then re-syncs your
  teams.
- The team sync, new connections and the lineup's default week now use the current NFL week
  from the schedule, not the week of the last finished refresh. A refresh that fails or is
  cut short no longer holds your teams on an old week.

Install `Fantasy-Manager-Setup-0.7.1.exe` over the previous version (0.7.0 has no updater, so
this one install is manual). Saved teams, the ESPN
profile and existing prediction history are retained. Your teams move to the current week on
the first refresh after the update.
