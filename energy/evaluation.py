# -*- coding: utf-8 -*-

import numpy as np
import theano
import theano.tensor as T

import logging

from sklearn import metrics
from sparse.learning import parse_embeddings


def auc_pr(predictions=[], labels=[]):
    '''Computes the Area Under the Precision-Recall Curve (AUC-PR)'''
    predictions, labels = np.asarray(predictions), np.asarray(labels)
    precision, recall, threshold = metrics.precision_recall_curve(labels, predictions)
    auc = metrics.auc(recall, precision)
    return auc

def auc_roc(predictions=[], labels=[]):
    '''Computes the Area Under the Receiver Operating Characteristic Curve (AUC-ROC)'''
    predictions, labels = np.asarray(predictions), np.asarray(labels)
    precision, recall, threshold = metrics.roc_curve(labels, predictions)
    auc = metrics.auc(recall, precision)
    return auc

#
# COMPUTING PERFORMANCE METRICS ON RANKINGS
#

#
# Evaluation summary (as in FB15k):
#
def ranking_summary(res, idxo=None, n=10, tag='raw'):
    resg = res[0] + res[1]
    dres = {}
    dres.update({'microlmean': np.mean(res[0])})
    dres.update({'microlmedian': np.median(res[0])})
    dres.update({'microlhits@n': np.mean(np.asarray(res[0]) <= n) * 100})
    dres.update({'micrormean': np.mean(res[1])})
    dres.update({'micrormedian': np.median(res[1])})
    dres.update({'microrhits@n': np.mean(np.asarray(res[1]) <= n) * 100})
    resg = res[0] + res[1]
    dres.update({'microgmean': np.mean(resg)})
    dres.update({'microgmedian': np.median(resg)})
    dres.update({'microghits@n': np.mean(np.asarray(resg) <= n) * 100})

    logging.info('### MICRO (%s):' % (tag))
    logging.info('\t-- left   >> mean: %s, median: %s, hits@%s: %s%%' % (
                    round(dres['microlmean'], 5), round(dres['microlmedian'], 5),
                    n, round(dres['microlhits@n'], 3)))
    logging.info('\t-- right  >> mean: %s, median: %s, hits@%s: %s%%' % (
                    round(dres['micrormean'], 5), round(dres['micrormedian'], 5),
                    n, round(dres['microrhits@n'], 3)))
    logging.info('\t-- global >> mean: %s, median: %s, hits@%s: %s%%' % (
                    round(dres['microgmean'], 5), round(dres['microgmedian'], 5),
                    n, round(dres['microghits@n'], 3)))

    if idxo is not None:
        listrel = set(idxo)
        dictrelres = {}
        dictrellmean = {}
        dictrelrmean = {}
        dictrelgmean = {}
        dictrellmedian = {}
        dictrelrmedian = {}
        dictrelgmedian = {}
        dictrellrn = {}
        dictrelrrn = {}
        dictrelgrn = {}

        for i in listrel:
            dictrelres.update({i: [[], []]})

        for i, j in enumerate(res[0]):
            dictrelres[idxo[i]][0] += [j]

        for i, j in enumerate(res[1]):
            dictrelres[idxo[i]][1] += [j]

        for i in listrel:
            dictrellmean[i] = np.mean(dictrelres[i][0])
            dictrelrmean[i] = np.mean(dictrelres[i][1])
            dictrelgmean[i] = np.mean(dictrelres[i][0] + dictrelres[i][1])
            dictrellmedian[i] = np.median(dictrelres[i][0])
            dictrelrmedian[i] = np.median(dictrelres[i][1])
            dictrelgmedian[i] = np.median(dictrelres[i][0] + dictrelres[i][1])
            dictrellrn[i] = np.mean(np.asarray(dictrelres[i][0]) <= n) * 100
            dictrelrrn[i] = np.mean(np.asarray(dictrelres[i][1]) <= n) * 100
            dictrelgrn[i] = np.mean(np.asarray(dictrelres[i][0] + dictrelres[i][1]) <= n) * 100

        dres.update({'dictrelres': dictrelres})
        dres.update({'dictrellmean': dictrellmean})
        dres.update({'dictrelrmean': dictrelrmean})
        dres.update({'dictrelgmean': dictrelgmean})
        dres.update({'dictrellmedian': dictrellmedian})
        dres.update({'dictrelrmedian': dictrelrmedian})
        dres.update({'dictrelgmedian': dictrelgmedian})
        dres.update({'dictrellrn': dictrellrn})
        dres.update({'dictrelrrn': dictrelrrn})
        dres.update({'dictrelgrn': dictrelgrn})

        dres.update({'macrolmean': np.mean(dictrellmean.values())})
        dres.update({'macrolmedian': np.mean(dictrellmedian.values())})
        dres.update({'macrolhits@n': np.mean(dictrellrn.values())})
        dres.update({'macrormean': np.mean(dictrelrmean.values())})
        dres.update({'macrormedian': np.mean(dictrelrmedian.values())})
        dres.update({'macrorhits@n': np.mean(dictrelrrn.values())})
        dres.update({'macrogmean': np.mean(dictrelgmean.values())})
        dres.update({'macrogmedian': np.mean(dictrelgmedian.values())})
        dres.update({'macroghits@n': np.mean(dictrelgrn.values())})

        logging.info('### MACRO (%s):' % (tag))
        logging.info('\t-- left   >> mean: %s, median: %s, hits@%s: %s%%' % (
                        round(dres['macrolmean'], 5), round(dres['macrolmedian'], 5),
                        n, round(dres['macrolhits@n'], 3)))
        logging.info('\t-- right  >> mean: %s, median: %s, hits@%s: %s%%' % (
                        round(dres['macrormean'], 5), round(dres['macrormedian'], 5),
                        n, round(dres['macrorhits@n'], 3)))
        logging.info('\t-- global >> mean: %s, median: %s, hits@%s: %s%%' % (
                        round(dres['macrogmean'], 5), round(dres['macrogmedian'], 5),
                        n, round(dres['macroghits@n'], 3)))

    return dres

#
# RANKING FUNCTIONS
#
def RankRightFnIdx(fnsim, embeddings, leftop, rightop, subtensorspec=None):
    """
    This function returns a Theano function to measure the similarity score of
    all 'right' entities given couples of relation and 'left' entities (as
    index values).

    :param fnsim: similarity function (on Theano variables).
    :param embeddings: an Embeddings instance.
    :param leftop: class for the 'left' operator.
    :param rightop: class for the 'right' operator.
    :param subtensorspec: only measure the similarity score for the entities
                          corresponding to the first subtensorspec (int)
                          entities of the embedding matrix (default None: all
                          entities).
    """
    embedding, relationl, relationr = parse_embeddings(embeddings)

    # Inputs
    idxl, idxo = T.iscalar('idxl'), T.iscalar('idxo')

    # Graph
    lhs = (embedding.E[:, idxl]).reshape((1, embedding.D))              # lhs: 1xD vector containing the embedding of idxl

    if subtensorspec is not None:
        # We compute the score only for a subset of entities
        rhs = (embedding.E[:, :subtensorspec]).T
    else:
        rhs = embedding.E.T                                             # rhs: NxD embedding matrix

    rell = (relationl.E[:, idxo]).reshape((1, relationl.D))             # rell: 1xD vector containing the embedding of idxo (relationl)
    relr = (relationr.E[:, idxo]).reshape((1, relationr.D))             # relr: 1xD vector containing the embedding of idxo (relationr)

    tmp = leftop(lhs, rell)                                             # a = rell(lhs)
                                                                        # b = relr(rhs)

    simi = fnsim(tmp.reshape((1, tmp.shape[1])), rightop(rhs, relr))    # simi = fnsim(a, b)
    """
    Theano function inputs.
    :input idxl: index value of the 'left' member.
    :input idxo: index value of the relation member.

    Theano function output.
    :output simi: vector of score values.
    """
    return theano.function([idxl, idxo], [simi], on_unused_input='ignore')

def RankLeftFnIdx(fnsim, embeddings, leftop, rightop, subtensorspec=None):
    """
    This function returns a Theano function to measure the similarity score of
    all 'left' entities given couples of relation and 'right' entities (as
    index values).

    :param fnsim: similarity function (on Theano variables).
    :param embeddings: an Embeddings instance.
    :param leftop: class for the 'left' operator.
    :param rightop: class for the 'right' operator.
    :param subtensorspec: only measure the similarity score for the entities
                          corresponding to the first subtensorspec (int)
                          entities of the embedding matrix (default None: all
                          entities).
    """
    embedding, relationl, relationr = parse_embeddings(embeddings)

    # Inputs
    idxr, idxo = T.iscalar('idxr'), T.iscalar('idxo')

    # Graph
    if subtensorspec is not None:
        # We compute the score only for a subset of entities
        lhs = (embedding.E[:, :subtensorspec]).T
    else:
        lhs = embedding.E.T

    rhs = (embedding.E[:, idxr]).reshape((1, embedding.D))
    rell = (relationl.E[:, idxo]).reshape((1, relationl.D))
    relr = (relationr.E[:, idxo]).reshape((1, relationr.D))

    tmp = rightop(rhs, relr)

    simi = fnsim(leftop(lhs, rell), tmp.reshape((1, tmp.shape[1])))
    """
    Theano function inputs.
    :input idxr: index value of the 'right' member.
    :input idxo: index value of the relation member.

    Theano function output.
    :output simi: vector of score values.
    """
    return theano.function([idxr, idxo], [simi], on_unused_input='ignore')

def RankingScoreIdx(sl, sr, idxl, idxr, idxo):
    """
    This function computes the rank list of the lhs and rhs, over a list of
    lhs, rhs and rel indexes.

    :param sl: Theano function created with RankLeftFnIdx().
    :param sr: Theano function created with RankRightFnIdx().
    :param idxl: list of 'left' indices.
    :param idxr: list of 'right' indices.
    :param idxo: list of relation indices.
    """
    errl = []
    errr = []

    for l, o, r in zip(idxl, idxo, idxr):
        errl += [np.argsort(np.argsort((sl(r, o)[0]).flatten())[::-1]).flatten()[l] + 1]
        errr += [np.argsort(np.argsort((sr(l, o)[0]).flatten())[::-1]).flatten()[r] + 1]

    return errl, errr

def FilteredRankingScoreIdx(sl, sr, idxl, idxr, idxo, true_triples):
    """
    This function computes the rank list of the lhs and rhs, over a list of
    lhs, rhs and rel indexes.
    :param sl: Theano function created with RankLeftFnIdx().
    :param sr: Theano function created with RankRightFnIdx().
    :param idxl: list of 'left' indices.
    :param idxr: list of 'right' indices.
    :param idxo: list of relation indices.
    """
    errl = []
    errr = []
    for l, o, r in zip(idxl, idxo, idxr):
        il=np.argwhere(true_triples[:,0]==l).reshape(-1,)
        io=np.argwhere(true_triples[:,1]==o).reshape(-1,)
        ir=np.argwhere(true_triples[:,2]==r).reshape(-1,)

        inter_l = [i for i in ir if i in io]
        rmv_idx_l = [true_triples[i,0] for i in inter_l if true_triples[i,0] != l]
        scores_l = (sl(r, o)[0]).flatten()
        scores_l[rmv_idx_l] = -np.inf
        errl += [np.argsort(np.argsort(-scores_l)).flatten()[l] + 1]

        inter_r = [i for i in il if i in io]
        rmv_idx_r = [true_triples[i,2] for i in inter_r if true_triples[i,2] != r]
        scores_r = (sr(l, o)[0]).flatten()
        scores_r[rmv_idx_r] = -np.inf
        errr += [np.argsort(np.argsort(-scores_r)).flatten()[r] + 1]
    return errl, errr

def RankingScoreIdx_sub(sl, sr, idxl, idxr, idxo, selection=[]):
    """
    Similar to RankingScoreIdx, but works on a subset of examples, defined in
    the 'selection' parameter.
    """
    errl, errr = [], []

    for l, o, r in [(idxl[i], idxo[i], idxr[i]) for i in selection]:
        errl += [np.argsort(np.argsort((sl(r, o)[0]).flatten())[::-1]).flatten()[l] + 1]
        errr += [np.argsort(np.argsort((sr(l, o)[0]).flatten())[::-1]).flatten()[r] + 1]

    return errl, errr

def RankingScoreRightIdx(sr, idxl, idxr, idxo):
    """
    This function computes the rank list of the rhs, over a list of lhs, rhs
    and rel indexes.

    :param sr: Theano function created with RankRightFnIdx().
    :param idxl: list of 'left' indices.
    :param idxr: list of 'right' indices.
    :param idxo: list of relation indices.
    """
    errr = []
    for l, o, r in zip(idxl, idxo, idxr):
        errr += [np.argsort(np.argsort((sr(l, o)[0]).flatten())[::-1]).flatten()[r] + 1]
    return errr

#
# COMPUTING PERFORMANCE METRICS ON CLASSIFICATIONS
#

def classification_summary(energyfn,
                            validlidx, validridx, validoidx, valid_targets,
                            testlidx, testridx, testoidx, test_targets):

    # Find unique relation indexes
    relidxs = np.unique(validoidx)

    valid_matches, test_matches = [], []

    # Iterate over unique relation indexes
    for relidx in relidxs:
        # Select the validation triples containing the 'relidx' predicate, and the corresponding target values
        valid_idxs = np.where(validoidx == relidx)
        r_validlidx, r_validridx, r_validoidx = validlidx[valid_idxs], validridx[valid_idxs], validoidx[valid_idxs]
        r_valid_targets = valid_targets[valid_idxs]

        # Evaluate the energies of those triples
        r_valid_energies = energyfn(r_validlidx, r_validridx, r_validoidx)[0]
        r_valid_cutpoint = find_classification_threshold(r_valid_energies, r_valid_targets)

        valid_matches += classification_matches(r_valid_energies, r_valid_targets, r_valid_cutpoint)

        # Select the test triples containing the 'relidx' predicate, and the corresponding target values
        test_idxs = np.where(testoidx == relidx)
        r_testlidx, r_testridx, r_testoidx = testlidx[test_idxs], testridx[test_idxs], testoidx[test_idxs]
        r_test_targets = test_targets[test_idxs]

        r_test_energies = energyfn(r_testlidx, r_testridx, r_testoidx)[0]
        test_matches += classification_matches(r_test_energies, r_test_targets, r_valid_cutpoint)

    logging.info('Validation Accuracy: %s -- Test Accuracy: %s' %
                    ((np.mean(valid_matches) * 100.0), (np.mean(test_matches) * 100.0)))

def find_classification_threshold(energies, targets):
    x = np.unique(np.sort(energies))
    cutpoints = np.concatenate(([x[0]], (x[1:] + x[:-1]) / 2., [x[-1]]))
    accuracies = [np.mean(classification_matches(energies, targets, cutpoint)) * 100.0 for cutpoint in cutpoints]
    best_cutpoint = cutpoints[np.argmax(np.asarray(accuracies))]
    return best_cutpoint

def classification_matches(energies, targets, threshold):
    classifications = classify(energies, threshold)
    comparisons = (targets == classifications)
    ret = [1. if comparison == True else 0. for comparison in comparisons]
    return ret

def classify(energies, threshold):
    classifications = np.asarray([1 if energy < threshold else 0 for energy in energies])
    return classifications

#
# CLASSIFICATION FUNCTIONS
#

def EnergyFn(fnsim, embeddings, leftop, rightop):
    embedding, relationl, relationr = parse_embeddings(embeddings)
    idxl, idxo, idxr = T.iscalar('idxl'), T.iscalar('idxo'), T.iscalar('idxr')
    lhs = (embedding.E[:, idxl]).reshape((1, embedding.D))
    rhs = (embedding.E[:, idxr]).reshape((1, embedding.D))
    rell = (relationl.E[:, idxo]).reshape((1, relationl.D))
    relr = (relationr.E[:, idxo]).reshape((1, relationr.D))
    energy = - fnsim(leftop(lhs, rell), rightop(rhs, relr))
    return theano.function([idxl, idxr, idxo], [energy], on_unused_input='ignore')

def EnergyVecFn(fnsim, embeddings, leftop, rightop):
    embedding, relationl, relationr = parse_embeddings(embeddings)
    idxl, idxo, idxr = T.ivector('idxl'), T.ivector('idxo'), T.ivector('idxr')
    lhs, rhs = embedding.E[:, idxl].T, embedding.E[:, idxr].T
    rell, relr = relationl.E[:, idxo].T, relationr.E[:, idxo].T
    energy = - fnsim(leftop(lhs, rell), rightop(rhs, relr))
    return theano.function([idxl, idxr, idxo], [energy], on_unused_input='ignore')

#
# LEVERAGING RANGE AND DOMAIN RELATIONS DURING LEARNING
#

def FilteredRankingScoreIdx_DR(sl, sr, idxl, idxr, idxo, rel2domain, rel2range, illegal_dr_penalty=1e6, true_triples=None):
    errl, errr = [], []

    relidxs = np.unique(idxo)
    for relidx in relidxs:

        dr_domain, dr_range = rel2domain[relidx], rel2range[relidx]

        dr_domain = set(dr_domain)
        dr_range = set(dr_range)

        test_triples = [(l, o, r) for (l, o, r) in zip(idxl, idxo, idxr) if o == relidx]
        for l, o, r in test_triples:

            rmv_idx_l, rmv_idx_r = [], []

            # Remove triples from true_triples from ranking results
            if true_triples is not None:
                il = np.argwhere(true_triples[:, 0] == l).reshape(-1,)
                io = np.argwhere(true_triples[:, 1] == o).reshape(-1,)
                ir = np.argwhere(true_triples[:, 2] == r).reshape(-1,)

                inter_l = [i for i in ir if i in io]
                rmv_idx_l += [true_triples[i, 0] for i in inter_l if true_triples[i, 0] != l]

                inter_r = [i for i in il if i in io]
                rmv_idx_r += [true_triples[i, 2] for i in inter_r if true_triples[i, 2] != r]

            scores_l = (sl(r, o)[0]).flatten()
            scores_r = (sr(l, o)[0]).flatten()

            # Remove triples not in domain and range from ranking results

            pen_idx_l = [cl for cl in range(len(scores_l)) if cl not in dr_domain]
            pen_idx_r = [cr for cr in range(len(scores_r)) if cr not in dr_range]

            scores_l[rmv_idx_l] = -np.inf
            scores_r[rmv_idx_r] = -np.inf

            scores_l[pen_idx_l] -= illegal_dr_penalty
            scores_r[pen_idx_r] -= illegal_dr_penalty

            errl += [np.argsort(np.argsort(- scores_l)).flatten()[l] + 1]
            errr += [np.argsort(np.argsort(- scores_r)).flatten()[r] + 1]

    return errl, errr

#
# SCHEMA-AWARE RANKING FUNCTIONS
#

#
# RANKING FUNCTIONS
#
def RankRightFnIdx_Schema(fnsim, embeddings, prior, leftop, rightop, subtensorspec=None):
    embedding, relationl, relationr = parse_embeddings(embeddings)

    # Inputs
    idxl, idxo = T.iscalar('idxl'), T.iscalar('idxo')
    g = T.matrix('g')

    # Graph
    lhs = (embedding.E[:, idxl]).reshape((1, embedding.D))              # lhs: 1xD vector containing the embedding of idxl

    if subtensorspec is not None:
        # We compute the score only for a subset of entities
        rhs = (embedding.E[:, :subtensorspec]).T
    else:
        rhs = embedding.E.T                                             # rhs: NxD embedding matrix

    rell = (relationl.E[:, idxo]).reshape((1, relationl.D))             # rell: 1xD vector containing the embedding of idxo (relationl)
    relr = (relationr.E[:, idxo]).reshape((1, relationr.D))             # relr: 1xD vector containing the embedding of idxo (relationr)

    tmp = leftop(lhs, rell)                                             # a = rell(lhs)
                                                                        # b = relr(rhs)

    # Negative Energy
    simi = fnsim(tmp.reshape((1, tmp.shape[1])), rightop(rhs, relr))    # simi = fnsim(a, b)

    pen_simi = g[0, :].T * prior.P[idxo, 0].T + g[1, :].T * prior.P[idxo, 1].T
    simi = simi - pen_simi

    return theano.function([idxl, idxo, g], [simi], on_unused_input='ignore')

def RankLeftFnIdx_Schema(fnsim, embeddings, prior, leftop, rightop, subtensorspec=None):
    embedding, relationl, relationr = parse_embeddings(embeddings)

    # Inputs
    idxr, idxo = T.iscalar('idxr'), T.iscalar('idxo')
    g = T.matrix('g')

    # Graph
    if subtensorspec is not None:
        # We compute the score only for a subset of entities
        lhs = (embedding.E[:, :subtensorspec]).T
    else:
        lhs = embedding.E.T

    rhs = (embedding.E[:, idxr]).reshape((1, embedding.D))
    rell = (relationl.E[:, idxo]).reshape((1, relationl.D))
    relr = (relationr.E[:, idxo]).reshape((1, relationr.D))

    tmp = rightop(rhs, relr)

    simi = fnsim(leftop(lhs, rell), tmp.reshape((1, tmp.shape[1])))

    pen_simi = g[0, :].T * prior.P[idxo, 0].T + g[1, :].T * prior.P[idxo, 1].T
    simi = simi - pen_simi

    return theano.function([idxr, idxo, g], [simi], on_unused_input='ignore')

#@profile
def RankingScoreIdx_Schema(sl, sr, idxl, idxr, idxo,
                            relation2domainSet, relation2rangeSet,
                            schemaPenalty, l_subtensorspec=None, r_subtensorspec=None):
    errl = []
    errr = []

    for l, o, r in zip(idxl, idxo, idxr):
        gl = schemaPenalty.schema_penalties_lr_fast(range(l_subtensorspec), [r] * l_subtensorspec, [o] * l_subtensorspec)
        gr = schemaPenalty.schema_penalties_lr_fast([l] * r_subtensorspec, range(r_subtensorspec), [o] * r_subtensorspec)

        slro = sl(r, o, gl)[0]
        srlo = sr(l, o, gr)[0]

        errl += [np.argsort(np.argsort((slro).flatten())[::-1]).flatten()[l] + 1]
        errr += [np.argsort(np.argsort((srlo).flatten())[::-1]).flatten()[r] + 1]

    return errl, errr

def FilteredRankingScoreIdx_Schema(sl, sr, idxl, idxr, idxo, true_triples,
                                    relation2domainSet, relation2rangeSet,
                                    schemaPenalty, l_subtensorspec=None, r_subtensorspec=None):
    errl = []
    errr = []

    for l, o, r in zip(idxl, idxo, idxr):
        il=np.argwhere(true_triples[:,0]==l).reshape(-1,)
        io=np.argwhere(true_triples[:,1]==o).reshape(-1,)
        ir=np.argwhere(true_triples[:,2]==r).reshape(-1,)

        gl = schemaPenalty.schema_penalties_lr_fast(range(l_subtensorspec), [r] * l_subtensorspec, [o] * l_subtensorspec)
        gr = schemaPenalty.schema_penalties_lr_fast([l] * r_subtensorspec, range(r_subtensorspec), [o] * r_subtensorspec)

        slro = sl(r, o, gl)[0]
        srlo = sr(l, o, gr)[0]

        inter_l = [i for i in ir if i in io]
        rmv_idx_l = [true_triples[i, 0] for i in inter_l if true_triples[i, 0] != l]
        scores_l = (slro).flatten()
        scores_l[rmv_idx_l] = -np.inf
        errl += [np.argsort(np.argsort(-scores_l)).flatten()[l] + 1]

        inter_r = [i for i in il if i in io]
        rmv_idx_r = [true_triples[i, 2] for i in inter_r if true_triples[i, 2] != r]
        scores_r = (srlo).flatten()
        scores_r[rmv_idx_r] = -np.inf
        errr += [np.argsort(np.argsort(-scores_r)).flatten()[r] + 1]
    return errl, errr
