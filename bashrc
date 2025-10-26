# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

set -o vi

alias lm="cd ~/scratch/colorlessgreenRNNs/src/language_models/"
alias povstim-with-childes="cd ~/scratch/lm-povstim-with-childes/"
alias data="cd ~/scratch/lm-povstim-with-childes/data"
alias src="cd ~/scratch/lm-povstim-with-childes/src"
alias models="cd ~/scratch/lm-povstim-with-childes/src/models"
alias lcrnn="cd ~/scratch/LCRNN/word-language-model/"
alias expand="cd ~/scratch/expand/LCRNN/word-language-model/"
#alias interact-command="echo \"interact -n 6 -p debug -g 1 -t 2:00:00\""
alias interact-run="interact -n 6 -p debug -g 1 -t 2:00:00"
alias interact-command="echo \"interact -n 6 -p debug -g 1 -t 2:00:00\""
alias lsrc="echo \"../scratch/lm-povstim-with-childes/src\""
alias evalparams="echo \"python eval.py --data ../data/CFG/linear.txt.data --finetuning_data ../data/CHILDES_final/finetuning/ --model models/LSTM_final/2-800-10-20-0.4-1001-LSTM-model.pt --rnn --cuda\""


vimsrc () { 
    cp "~/scratch/lm-povstim-with-childes/src/$1" .
    vim "$1"
    cp "$1 ~/scratch/lm-povstim-with-childes/src/" 
}

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

# User specific aliases and functions

# # >>> conda initialize >>>
# # !! Contents within this block are managed by 'conda init' !!
# __conda_setup="$('/software/apps/anaconda/2019.03/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
# if [ $? -eq 0 ]; then
#     eval "$__conda_setup"
# else
#     if [ -f "/software/apps/anaconda/2019.03/etc/profile.d/conda.sh" ]; then
#         . "/software/apps/anaconda/2019.03/etc/profile.d/conda.sh"
#     else
#         export PATH="/software/apps/anaconda/2019.03/bin:$PATH"
#     fi
# fi
# unset __conda_setup
# # <<< conda initialize <<<

#conda activate
#. /software/apps/anaconda/5.2/python/3.6/etc/profile.d/conda.sh
