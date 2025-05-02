# 1. SSH in
ssh yihan@phoenix

# 2. Create / attach a session called "tune"
tmux new -s tune          # or: tmux attach -t tune

# 3. Inside tmux, launch the sweep
PYTHONPATH=. python cli/tune_nrms_optuna.py --trials 50 --gpus 4

# 4. Detach at any time
Ctrl-b  d

# 5. Re-attach to the session
tmux attach -t tune

# 6. kill the session
tmux kill-session -t tune

# 7. List all sessions
tmux list-sessions