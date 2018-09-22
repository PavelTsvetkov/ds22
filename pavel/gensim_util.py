import gensim

mdl = gensim.models.KeyedVectors.load_word2vec_format("C:\\tmp\\dabble\\GoogleNews-vectors-negative300.bin", binary=True)

mdl.save("C:\\tmp\\dabble\\GoogleNews-kvectors.bin")