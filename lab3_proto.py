import numpy as np

from lab2_tools import *
from lab2_proto import *

from lab3_tools import *



def words2phones(wordList, pronDict, addSilence=True, addShortPause=True):
      """ word2phones: converts word level to phone level transcription adding silence

      Args:
         wordList: list of word symbols
         pronDict: pronunciation dictionary. The keys correspond to words in wordList
         addSilence: if True, add initial and final silence
         addShortPause: if True, add short pause model "sp" at end of each word
      Output:
         list of phone symbols
      """
      phones = []
      if addSilence:
         phones.append('sil')
      for (idx, word) in enumerate(wordList):
         phones.extend(pronDict[word])
         if addShortPause and idx < len(wordList) - 1:
            phones.append('sp')
      if addSilence:
         phones.append('sil')
      return phones


def forcedAlignment(lmfcc, phoneHMMs, phoneTrans):
      """ forcedAlignmen: aligns a phonetic transcription at the state level

      Args:
         lmfcc: NxD array of MFCC feature vectors (N vectors of dimension D)
               computed the same way as for the training of phoneHMMs
         phoneHMMs: set of phonetic Gaussian HMM models
         phoneTrans: list of phonetic symbols to be aligned including initial and
                     final silence

      Returns:
         list of strings in the form phoneme_index specifying, for each time step
         the state from phoneHMMs corresponding to the viterbi path.
      """
      # build the combined model and the per-state name list
      utteranceHMM = concatHMMs(phoneHMMs, phoneTrans)

      # ??? modelling of skip transition, leaving out silence pause (match-patch)
      transmat = utteranceHMM['transmat'].copy()
      idx = 0
      for k, ph in enumerate(phoneTrans):
         if ph == 'sp' and 0 < k < len(phoneTrans) - 1:
            before, sp0, after = idx - 1, idx, idx + 1
            # move heuristically probability mass from "before -> sp_0" into "before -> after"
            skip_mass = transmat[before, sp0] * 0.88           
            transmat[before, after] = skip_mass
            transmat[before, sp0]  -= skip_mass
         idx += phoneHMMs[ph]['means'].shape[0]
      utteranceHMM['transmat'] = transmat

      nr_states = {phone_model: phoneHMMs[phone_model]['means'].shape[0] for phone_model in phoneHMMs}
      stateTrans = [phone + '_' + str(state_idx)
                  for phone in phoneTrans
                  for state_idx in range(nr_states[phone])]

      S = len(stateTrans)
      means  = utteranceHMM['means'][:S]
      covars = utteranceHMM['covars'][:S]

      # computation of obversation log-likelihood for Viterbi algorithm
      obsloglik = log_multivariate_normal_density_diag(lmfcc, means, covars)

      # Viterbi on log-domain transitions
      log_startprob = np.log(utteranceHMM['startprob'][:S])
      log_transmat  = np.log(utteranceHMM['transmat'][:S, :S])
      _, viterbiPath = viterbi(obsloglik, log_startprob, log_transmat)

      # convert path indices to phone-state names
      viterbiStateTrans = [stateTrans[i] for i in viterbiPath]

      return viterbiStateTrans


