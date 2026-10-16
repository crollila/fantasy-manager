; Updates install over a running app: end the bundled engine too, or its files stay locked.
!macro customInit
  nsExec::Exec 'taskkill /F /T /IM FantasyBackend.exe'
!macroend
