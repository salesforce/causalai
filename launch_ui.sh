#!/bin/zsh
flask --app ui_backend run &
cd ui
npm install
npm run dev