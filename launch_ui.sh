#!/bin/zsh
cd ui
flask --app ui_backend run &
npm install
npm run dev &
cd ..
exec zsh