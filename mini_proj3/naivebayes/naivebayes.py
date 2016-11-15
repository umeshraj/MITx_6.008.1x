import sys
import os.path
import numpy as np
from collections import Counter, defaultdict

import util

USAGE = "%s <test data folder> <spam folder> <ham folder>"

def get_counts(file_list):
    """
    Computes counts for each word that occurs in the files in file_list.

    Inputs
    ------
    file_list : a list of filenames, suitable for use with open() or
                util.get_words_in_file()

    Output
    ------
    A dict whose keys are words, and whose values are the number of files the
    key occurred in.
    """
    ### TODO: Comment out the following line and write your code here
    word_count = Counter()
    for file in file_list:
        word_list = util.get_words_in_file(file)
        word_list = set(word_list)  # ignore duplicates
        for word in word_list:
            word_count[word] += 1
    return word_count


def get_log_probabilities(file_list):
    """
    Computes log-frequencies for each word that occurs in the files in
    file_list.

    Input
    -----
    file_list : a list of filenames, suitable for use with open() or
                util.get_words_in_file()

    Output
    ------
    A dict whose keys are words, and whose values are the log of the smoothed
    estimate of the fraction of files the key occurred in.

    Hint
    ----
    The data structure util.DefaultDict will be useful to you here, as will the
    get_counts() helper above.
    """
    ### TODO: Comment out the following line and write your code here
    word_count = get_counts(file_list)
    num_files = len(file_list)
    ret = np.log(1/(num_files+2))
    log_prob =  defaultdict(lambda: ret)
    #log_prob = defaultdict(lambda: 0)
    extra_num = 1
    extra_den = 2
    for word, num_words in word_count.items():
        log_prob[word] = np.log((num_words+extra_num)/(num_files+extra_den))
    return log_prob


def learn_distributions(file_lists_by_category):
    """
    Input
    -----
    A two-element list. The first element is a list of spam files,
    and the second element is a list of ham (non-spam) files.

    Output
    ------
    (log_probabilities_by_category, log_prior)

    log_probabilities_by_category : A list whose first element is a smoothed
                                    estimate for log P(y=w_j|c=spam) (as a dict,
                                    just as in get_log_probabilities above), and
                                    whose second element is the same for c=ham.

    log_prior_by_category : A list of estimates for the log-probabilities for
                            each class:
                            [est. for log P(c=spam), est. for log P(c=ham)]
    """
    ### TODO: Comment out the following line and write your code here
    #raise NotImplementedError
    log_prob_by_category = []
    for file_list_catergory in file_lists_by_category:
        tmp = get_log_probabilities(file_list_catergory)
        log_prob_by_category.append(tmp)

    # compute the log_prior
    num_spam = len(file_lists_by_category[0])
    num_ham = len(file_lists_by_category[1])
    num_files = num_spam + num_ham

    log_spam = np.log((num_spam+0)/(num_files+0))
    log_ham = np.log((num_ham+0)/(num_files+0))
    log_prior_by_category = [log_spam, log_ham]

    return log_prob_by_category, log_prior_by_category


def classify_email(email_filename,
                   log_probabilities_by_category,
                   log_prior_by_category):
    """
    Uses Naive Bayes classification to classify the email in the given file.

    Inputs
    ------
    email_filename : name of the file containing the email to be classified

    log_probabilities_by_category : See output of learn_distributions

    log_prior_by_category : See output of learn_distributions

    Output
    ------
    One of the labels in names.
    """
    ### TODO: Comment out the following line and write your code here
    log_spam = log_prior_by_category[0]
    log_ham = log_prior_by_category[1]
    log_prob_spam = log_probabilities_by_category[0]
    log_prob_ham = log_probabilities_by_category[1]

    # compute posteriors
    word_list = util.get_words_in_file(email_filename)
    #word_list = set(word_list)

    post_spam = log_spam
    post_ham = log_ham
    for word in word_list:
        post_spam += log_prob_spam[word]
        post_ham += log_prob_ham[word]
#    post_spam = log_spam
#    for spam_word, spam_prob in log_prob_spam.items():
#        if spam_word in word_list:
#            post_spam += spam_prob
#        else:
#            post_spam += np.log(1-np.exp(spam_prob))
#
#    post_ham = log_ham
#    for ham_word, ham_prob in log_prob_ham.items():
#        if ham_word in word_list:
#            post_ham += ham_prob
#        else:
#            post_ham += np.log(1-np.exp(ham_prob))

    class_pred = 'spam' if post_spam >= post_ham else 'ham'

    return class_pred

def classify_emails(spam_files, ham_files, test_files):
    # DO NOT MODIFY -- used by the autograder
    log_probabilities_by_category, log_prior = \
        learn_distributions([spam_files, ham_files])
    estimated_labels = []
    for test_file in test_files:
        estimated_label = \
            classify_email(test_file, log_probabilities_by_category, log_prior)
        estimated_labels.append(estimated_label)
    return estimated_labels

def main():
    ### Read arguments
#    if len(sys.argv) != 4:
#        print(USAGE % sys.argv[0])
#    testing_folder = sys.argv[1]
#    (spam_folder, ham_folder) = sys.argv[2:4]
    testing_folder = 'data_two/testing/'
    spam_folder = 'data_two/spam/'
    ham_folder = 'data_two/ham/'

    ### Learn the distributions
    file_lists = []
    for folder in (spam_folder, ham_folder):
        file_lists.append(util.get_files_in_folder(folder))
    (log_probabilities_by_category, log_priors_by_category) = \
            learn_distributions(file_lists)

    # Here, columns and rows are indexed by 0 = 'spam' and 1 = 'ham'
    # rows correspond to true label, columns correspond to guessed label
    performance_measures = np.zeros([2,2])

    ### Classify and measure performance
    for filename in (util.get_files_in_folder(testing_folder)):
        ## Classify
        label = classify_email(filename,
                               log_probabilities_by_category,
                               log_priors_by_category)
        ## Measure performance
        # Use the filename to determine the true label
        base = os.path.basename(filename)
        true_index = ('ham' in base)
        guessed_index = (label == 'ham')
        performance_measures[true_index, guessed_index] += 1


        # Uncomment this line to see which files your classifier
        # gets right/wrong:
        #print("%s : %s" %(label, filename))

    template="You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
    # Correct counts are on the diagonal
    correct = np.diag(performance_measures)
    # totals are obtained by summing across guessed labels
    totals = np.sum(performance_measures, 1)
    print(template % (correct[0],
                      totals[0],
                      correct[1],
                      totals[1]))
    print(performance_measures)

if __name__ == '__main__':
    main()
#    ur_counts = get_counts(['word_test.txt', 'ham_test.txt', 'spam_test.txt'])
#    ur_log_prob = get_log_probabilities(['word_test.txt'])
#    file_lists = [['spam_test.txt'], ['ham_test.txt']]
#    (log_probabilities_by_category, log_priors_by_category) = \
#            learn_distributions(file_lists)
#    class_predicted = classify_email('new_test.txt',
#                                     log_probabilities_by_category,
#                                     log_priors_by_category)