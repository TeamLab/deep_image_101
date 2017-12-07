import model

text_cnn = model.ModellingCnn(batch_size=64, embedding_size=100, first_filter=7,
                              second_filter=5, top_k=4, total_layer=3,
                              first_featuremap=8, second_featuremap =5, training_epochs=300)

text_cnn.train()
