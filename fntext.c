#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <assert.h>
#include <omp.h>

#define EM_RANGE (0.01)
typedef float floatx;

struct model_t
{
    floatx *em, *w, *b;
    int64_t em_dim, vocab_num, category_num;
};

struct dataset_t
{
    int64_t *text_indices, *text_lens, *text_categories;
    int64_t *start_pos;
    int64_t text_num;
};

void init_model(struct model_t *model, int64_t em_dim, int64_t vocab_num, int64_t category_num, int64_t is_init)
{
    model->em_dim = em_dim;
    model->vocab_num = vocab_num;
    model->category_num = category_num;

    model->em = (floatx *)malloc(em_dim * vocab_num * sizeof(floatx));
    model->w = (floatx *)malloc(em_dim * category_num * sizeof(floatx));
    model->b = (floatx *)malloc(category_num * sizeof(floatx));

    floatx *em = model->em;
    floatx *w = model->w;
    floatx *b = model->b;
    int64_t i;
    if (is_init)
    {
        srand(time(NULL));
        // [-EM_RANGE, EM_RANGE]
        for (i = 0; i < em_dim * vocab_num; i++)
            em[i] = ((floatx)rand() / RAND_MAX) * 2. * EM_RANGE - EM_RANGE;
        floatx stdv = 1. / (floatx)sqrt((double)em_dim);
        for (i = 0; i < em_dim * category_num; i++)
            w[i] = (floatx)rand() / RAND_MAX * 2. * stdv - stdv;
        for (i = 0; i < category_num; i++)
            b[i] = (floatx)rand() / RAND_MAX * 2. * stdv - stdv;
    }
    else
    {
        for (i = 0; i < em_dim * vocab_num; i++)
            em[i] = 0.;
        for (i = 0; i < em_dim * category_num; i++)
            w[i] = 0.;
        for (i = 0; i < category_num; i++)
            b[i] = 0.;
    }
}
void free_model(struct model_t *model)
{
    free(model->em);
    free(model->w);
    free(model->b);
}

int preread(FILE *fp)
{
    int ch = fgetc(fp);
    if (ch == EOF)
        return ch;
    else
    {
        fseek(fp, -1, SEEK_CUR);
        return ch;
    }
}
void load_data(struct dataset_t *data, const char *path, int64_t max_voc)
{
    FILE *fp = NULL;
    fp = fopen(path, "r");
    if (fp == NULL)
    {
        perror("error");
        exit(EXIT_FAILURE);
    }
    int next_i, next_ch;
    int64_t text_num = 0, ch_num = 0, ignore_text_num = 0;
    int64_t text_len = 0;
    ;
    int64_t cat, text_i;
    enum state_t
    {
        READ_CAT,
        READ_INDEX
    } state = READ_CAT;
    while (1)
    {
        int is_break = 0;
        switch (state)
        {
        case READ_CAT:
            if (fscanf(fp, "%ld,", &cat) > 0)
            {
                if (preread(fp) == '\n')
                {
                    ignore_text_num++;
                    fgetc(fp);
                }
                else
                    state = READ_INDEX;
            }
            else
            {
                assert(feof(fp));
                is_break = 1;
            }
            break;
        case READ_INDEX:
            assert(fscanf(fp, "%ld", &text_i) > 0);
            if (text_i < max_voc)
            {
                ch_num++;
                text_len++;
            }
            next_ch = fgetc(fp);
            if (next_ch == '\n')
            {
                if (text_len == 0)
                {
                    ignore_text_num++;
                }
                else
                {
                    text_num++;
                    text_len = 0;
                }
                state = READ_CAT;
            }
        }
        if (is_break)
            break;
    }
    printf("load data from %s\n", path);
    printf("#lines: %ld, #chs: %ld\n", text_num, ch_num);
    printf("#ignore lines: %ld\n", ignore_text_num);
    data->text_num = text_num;
    data->text_indices = (int64_t *)malloc(ch_num * sizeof(int64_t));
    data->text_lens = (int64_t *)malloc(text_num * sizeof(int64_t));
    data->text_categories = (int64_t *)malloc(text_num * sizeof(int64_t));
    data->start_pos = (int64_t *)malloc(text_num * sizeof(int64_t));

    text_len = 0;
    int64_t *text_indices = data->text_indices;
    int64_t *text_lens = data->text_lens;
    int64_t *text_categories = data->text_categories;
    int64_t *start_pos = data->start_pos;
    rewind(fp);
    while (1)
    {
        int is_break = 0;
        switch (state)
        {
        case READ_CAT:
            if (fscanf(fp, "%ld,", &cat) > 0)
            {
                if (preread(fp) == '\n')
                {
                    fgetc(fp);
                }
                else
                    state = READ_INDEX;
            }
            else
            {
                assert(feof(fp));
                is_break = 1;
            }
            break;
        case READ_INDEX:
            assert(fscanf(fp, "%ld", &text_i) > 0);
            if (text_i < max_voc)
            {
                text_len++;
                *text_indices = text_i;
                text_indices++;
            }
            next_ch = fgetc(fp);
            if (next_ch == '\n')
            {
                state = READ_CAT;
                if (text_len > 0)
                {
                    *text_lens = text_len;
                    text_lens++;
                    text_len = 0;

                    *text_categories = cat;
                    text_categories++;
                }
            }
        }
        if (is_break)
            break;
    }
    start_pos[0] = 0;
    for (int64_t i = 1; i < text_num; i++)
        start_pos[i] = start_pos[i - 1] + data->text_lens[i - 1];
    fclose(fp);
}



void free_data(struct dataset_t *data)
{
    free(data->text_indices);
    free(data->text_lens);
    free(data->text_categories);
    free(data->start_pos);
}

floatx forward(struct model_t *model, struct dataset_t *train_data, int64_t text_i, floatx *max_fea, int64_t *max_fea_index, floatx *softmax_fea)
{
    int64_t *text_indices = &train_data->text_indices[train_data->start_pos[text_i]];
    int64_t text_len = train_data->text_lens[text_i];
    assert(text_len >= 1);
    int64_t text_category = train_data->text_categories[text_i];

    int64_t i, j;
    int64_t em_pos;

    // max_pool
    // 先赋预值
    em_pos = text_indices[0] * model->em_dim;
    for (i = 0; i < model->em_dim; i++)
    {
        max_fea[i] = model->em[em_pos + i];
        max_fea_index[i] = em_pos + i;
    }

    floatx fea;
    int64_t fea_i;
    for (i = 1; i < text_len; i++)
    {
        em_pos = text_indices[i] * model->em_dim;
        for (j = 0; j < model->em_dim; j++)
        {
            max_fea[j] = max_fea[j] > (model->em[em_pos + j]) ? max_fea[j] : (model->em[em_pos + j]);
            max_fea_index[j] = max_fea[j] > (model->em[em_pos + j]) ? max_fea_index[j] : (em_pos + j);
        }
    }

    // mlp
    for (i = 0; i < model->category_num; i++)
        softmax_fea[i] = model->b[i];

    for (i = 0; i < model->category_num; i++)
        for (j = 0; j < model->em_dim; j++)
            softmax_fea[i] += max_fea[j] * model->w[i * model->em_dim + j];

    floatx loss = 0.;
    floatx tmp = 0.;
    loss -= softmax_fea[text_category];
    for (i = 0; i < model->category_num; i++)
    {
        softmax_fea[i] = (floatx)exp((double)softmax_fea[i]);
        tmp += softmax_fea[i];
    }
    loss += (floatx)log(tmp);
    return loss;
}

void backward(struct model_t *model, struct dataset_t *train_data, int64_t text_i, floatx *max_fea, floatx *softmax_fea, floatx *grad_em, floatx *grad_w, floatx *grad_b)
{
    int64_t *text_indices = &(train_data->text_indices[train_data->start_pos[text_i]]);
    int64_t text_len = train_data->text_lens[text_i];
    int64_t text_category = train_data->text_categories[text_i];

    floatx tmp_sum = 0.;
    int64_t i, j;
    for (i = 0; i < model->category_num; i++)
        tmp_sum += softmax_fea[i];
    for (i = 0; i < model->category_num; i++)
        grad_b[i] = softmax_fea[i] / tmp_sum;
    grad_b[text_category] -= 1.;

    for (i = 0; i < model->category_num; i++)
        for (j = 0; j < model->em_dim; j++)
            grad_w[i * model->em_dim + j] = max_fea[j] * grad_b[i];

    for (j = 0; j < model->em_dim; j++)
        grad_em[j] = 0.;
    for (i = 0; i < model->category_num; i++)
        for (j = 0; j < model->em_dim; j++)
            grad_em[j] += (model->w[i * model->em_dim + j]) * grad_b[i];
}

void evaluate(struct model_t *model, struct dataset_t *vali_data, int64_t batch_size, int64_t threads_n)
{
    printf("evaluating...\n");

    time_t eva_start, eva_end;
    eva_start = time(NULL);

    floatx *max_feas = (floatx *)malloc(model->em_dim * batch_size * sizeof(floatx));
    int64_t *max_fea_indexs = (int64_t *)malloc(model->em_dim * batch_size * sizeof(int64_t));
    floatx *softmax_feas = (floatx *)malloc(model->category_num * batch_size * sizeof(floatx));
    int64_t *pre_labels = (int64_t *)malloc(batch_size * sizeof(int64_t));
    int64_t *real_labels = (int64_t *)malloc(batch_size * sizeof(int64_t));
    // 临界资源
    floatx *cat_all = (floatx *)malloc(model->category_num * sizeof(floatx));
    floatx *cat_true = (floatx *)malloc(model->category_num * sizeof(floatx));

    for (int64_t i = 0; i < model->category_num; i++)
    {
        cat_all[i] = 0.;
        cat_true[i] = 0.;
    }

    for (int64_t batch_i = 0; batch_i < (vali_data->text_num + batch_size - 1) / batch_size; batch_i++)
    {
        int64_t real_batch_size = (vali_data->text_num - batch_i * batch_size) > batch_size ? batch_size : (vali_data->text_num - batch_i * batch_size);
#pragma omp parallel for schedule(dynamic) num_threads(threads_n)
        for (int64_t batch_j = 0; batch_j < real_batch_size; batch_j++)
        {
            int64_t text_i = (batch_i)*batch_size + batch_j;
            assert(text_i < vali_data->text_num);

            int64_t text_category = vali_data->text_categories[text_i];
            // 长度为0的text，不计算梯度
            // 会导致问题，比如梯度没有更新
            // 应该在生成数据时避免
            if (vali_data->text_lens[text_i] == 0)
            {
                printf("error: vali text length can not be zero.[text id: %ld]", text_i);
                exit(-1);
            }

            floatx *max_fea = &max_feas[batch_j * model->em_dim];
            int64_t *max_fea_index = &max_fea_indexs[batch_j * model->em_dim];
            floatx *softmax_fea = &softmax_feas[batch_j * model->category_num];
            int64_t *pre_label = &pre_labels[batch_j];
            int64_t *real_label = &real_labels[batch_j];

            *real_label = text_category;

            forward(model, vali_data, text_i, max_fea, max_fea_index, softmax_fea);
            *pre_label = 0;
            floatx fea = softmax_fea[0];
            for (int64_t c = 1; c < model->category_num; c++)
            {
                if (softmax_fea[c] > fea)
                {
                    *pre_label = c;
                    fea = softmax_fea[c];
                }
            }
        }

        // 访问临界资源
        for (int64_t batch_j = 0; batch_j < real_batch_size; batch_j++)
        {
            cat_all[real_labels[batch_j]] += 1;
            if (real_labels[batch_j] == pre_labels[batch_j])
                cat_true[real_labels[batch_j]] += 1;
        }
    }
    floatx cat_all_sum = 0.;
    floatx cat_true_sum = 0.;
    for (int64_t k = 0; k < model->category_num; k++)
    {
        cat_all_sum += cat_all[k];
        cat_true_sum += cat_true[k];
    }
    printf("#samples: %.0f\n", cat_all_sum);
    printf("macro precision: %.5f\n", cat_true_sum / cat_all_sum);
    for (int64_t k = 0; k < model->category_num; k++)
        printf("   category #%ld precision: %.5f\n", k, cat_true[k] / cat_all[k]);

    free(max_feas);
    free(max_fea_indexs);
    free(softmax_feas);
    free(pre_labels);
    free(real_labels);
    free(cat_all);
    free(cat_true);

    eva_end = time(NULL);
    printf("   evaluating time: %lds\n", eva_end-eva_start);
}

void train_sgd(struct model_t *model, struct dataset_t *train_data, struct dataset_t *vali_data, int64_t epochs, int64_t batch_size, floatx lr, int64_t threads_n)
{
    int64_t *shuffle_index = (int64_t *)malloc(train_data->text_num * sizeof(int64_t));
    int64_t tmp, i, sel;

    floatx *grads_em = (floatx *)malloc(model->em_dim * batch_size * sizeof(floatx));
    floatx *grads_w = (floatx *)malloc(model->em_dim * model->category_num * batch_size * sizeof(floatx));
    floatx *grads_b = (floatx *)malloc(model->category_num * batch_size * sizeof(floatx));

    floatx *max_feas = (floatx *)malloc(model->em_dim * batch_size * sizeof(floatx));
    int64_t *max_fea_indexs = (int64_t *)malloc(model->em_dim * batch_size * sizeof(int64_t));
    floatx *softmax_feas = (floatx *)malloc(model->category_num * batch_size * sizeof(floatx));
    floatx *losses = (floatx *)malloc(batch_size * sizeof(floatx));

    for (i = 0; i < train_data->text_num; i++)
        shuffle_index[i] = i;

    for (int64_t epoch = 0; epoch < epochs; epoch++)
    {
        printf("#epoch: %ld\n", epoch);
        floatx s_loss = 0.;
        time_t epoch_start, epoch_end;
        // shuffle
        for (i = 0; i < train_data->text_num; i++)
        {
            sel = rand() % (train_data->text_num - i) + i;
            tmp = shuffle_index[i];
            shuffle_index[i] = shuffle_index[sel];
            shuffle_index[sel] = tmp;
        }

        /* for (i=0; i<model->em_dim*batch_size; i++) grads_em[i] = 0.; */
        /* for (i=0; i<model->em_dim*model->category_num*batch_size; i++) grads_w[i] = 0.; */
        /* for (i=0; i<model->category_num->batch_size; i++) grads_b = 0.; */

        epoch_start = time(NULL);
        for (int64_t batch_i = 0; batch_i < (train_data->text_num + batch_size - 1) / batch_size; batch_i++)
        {
            int64_t real_batch_size = (train_data->text_num - batch_i * batch_size) > batch_size ? batch_size : (train_data->text_num - batch_i * batch_size);
            // 可以加速
#pragma omp parallel for schedule(dynamic) num_threads(threads_n)
            for (int64_t batch_j = 0; batch_j < real_batch_size; batch_j++)
            {
                int64_t text_i = (batch_i)*batch_size + batch_j;
                assert(text_i < train_data->text_num);
                text_i = shuffle_index[text_i];

                // 长度为0的text，不计算梯度
                // 会导致问题，比如梯度没有更新
                // 应该在生成数据时避免
                if (train_data->text_lens[text_i] == 0)
                {
                    printf("error: training text length can not be zero.[text id: %ld]", text_i);
                    exit(-1);
                }

                floatx *grad_em = &grads_em[batch_j * model->em_dim];
                floatx *grad_w = &grads_w[batch_j * model->em_dim * model->category_num];
                floatx *grad_b = &grads_b[batch_j * model->category_num];

                floatx *max_fea = &max_feas[batch_j * model->em_dim];
                int64_t *max_fea_index = &max_fea_indexs[batch_j * model->em_dim];
                floatx *softmax_fea = &softmax_feas[batch_j * model->category_num];

                losses[batch_j] = forward(model, train_data, text_i, max_fea, max_fea_index, softmax_fea);
                backward(model, train_data, text_i, max_fea, softmax_fea, grad_em, grad_w, grad_b);
            }

            for (int64_t batch_j = 0; batch_j < real_batch_size; batch_j++)
                s_loss += losses[batch_j];
            // update param
            for (int64_t batch_j = 0; batch_j < real_batch_size; batch_j++)
            {
                floatx *grad_em = &grads_em[batch_j * model->em_dim];
                floatx *grad_w = &grads_w[batch_j * model->em_dim * model->category_num];
                floatx *grad_b = &grads_b[batch_j * model->category_num];

                int64_t *max_fea_index = &max_fea_indexs[batch_j * model->em_dim];

                for (int64_t w_j = 0; w_j < model->em_dim * model->category_num; w_j++)
                    model->w[w_j] -= (lr * (1. / real_batch_size) * grad_w[w_j]);
                for (int64_t b_j = 0; b_j < model->category_num; b_j++)
                    model->b[b_j] -= (lr * (1. / real_batch_size) * grad_b[b_j]);
                for (int64_t em_j = 0; em_j < model->em_dim; em_j++)
                    model->em[max_fea_index[em_j]] -= (lr * (1. / real_batch_size) * grad_em[em_j]);
                /* model->em[max_fea_index[em_j]] -= (lr*grad_em[em_j]); */
            }

        } // end_batch
        epoch_end = time(NULL);
        s_loss /= train_data->text_num;
        printf("    loss: %.4f\n", s_loss);
        printf("    time: %lds\n", epoch_end - epoch_start);

        if (vali_data != NULL)
        {
            printf("evaluate vali data...\n");
            evaluate(model, vali_data, batch_size, threads_n);
        }
            
        printf("\n");

    } //end_epoch
    free(shuffle_index);
    free(grads_em);
    free(grads_w);
    free(grads_b);
    free(max_feas);
    free(max_fea_indexs);
    free(softmax_feas);
    free(losses);
}

void train_adam(struct model_t *model, struct dataset_t *train_data, struct dataset_t *vali_data, int64_t epochs, int64_t batch_size, int64_t threads_n)
{
    printf("start training(Adam)...\n");
    // omp_lock_t omplock;
    // omp_init_lock(&omplock);

    int64_t tmp, i, sel;

    floatx alpha = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8;
    floatx beta1t = beta1;
    floatx beta2t = beta2;

    int64_t *shuffle_index = (int64_t *)malloc(train_data->text_num * sizeof(int64_t));

    struct model_t adam_m, adam_v, gt;
    init_model(&adam_m, model->em_dim, model->vocab_num, model->category_num, 0);
    init_model(&adam_v, model->em_dim, model->vocab_num, model->category_num, 0);
    init_model(&gt, model->em_dim, model->vocab_num, model->category_num, 0);

    floatx *grads_em = (floatx *)malloc(model->em_dim * batch_size * sizeof(floatx));
    floatx *grads_w = (floatx *)malloc(model->em_dim * model->category_num * batch_size * sizeof(floatx));
    floatx *grads_b = (floatx *)malloc(model->category_num * batch_size * sizeof(floatx));

    floatx *max_feas = (floatx *)malloc(model->em_dim * batch_size * sizeof(floatx));
    int64_t *max_fea_indexs = (int64_t *)malloc(model->em_dim * batch_size * sizeof(int64_t));
    floatx *softmax_feas = (floatx *)malloc(model->category_num * batch_size * sizeof(floatx));
    floatx *losses = (floatx *)malloc(batch_size * sizeof(floatx));

    printf("init grad end...\n");

    for (i = 0; i < train_data->text_num; i++)
        shuffle_index[i] = i;

    for (int64_t epoch = 0; epoch < epochs; epoch++)
    {
        printf("#epoch: %ld\n", epoch);
        floatx s_loss = 0.;
        time_t epoch_start, epoch_end;
        // shuffle
        for (i = 0; i < train_data->text_num; i++)
        {
            sel = rand() % (train_data->text_num - i) + i;
            tmp = shuffle_index[i];
            shuffle_index[i] = shuffle_index[sel];
            shuffle_index[sel] = tmp;
        }

        epoch_start = time(NULL);
        for (int64_t batch_i = 0; batch_i < (train_data->text_num + batch_size - 1) / batch_size; batch_i++)
        {
            int64_t real_batch_size = (train_data->text_num - batch_i * batch_size) > batch_size ? batch_size : (train_data->text_num - batch_i * batch_size);
            // 可以加速
#pragma omp parallel for schedule(dynamic) num_threads(threads_n)
            for (int64_t batch_j = 0; batch_j < real_batch_size; batch_j++)
            {
                int64_t text_i = (batch_i)*batch_size + batch_j;
                assert(text_i < train_data->text_num);
                text_i = shuffle_index[text_i];

                // 长度为0的text，不计算梯度
                // 会导致问题，比如梯度没有更新
                // 应该在生成数据时避免
                if (train_data->text_lens[text_i] == 0)
                {
                    printf("error: training text length can not be zero.[text id: %ld]", text_i);
                    exit(-1);
                }

                floatx *grad_em = &grads_em[batch_j * model->em_dim];
                floatx *grad_w = &grads_w[batch_j * model->em_dim * model->category_num];
                floatx *grad_b = &grads_b[batch_j * model->category_num];

                floatx *max_fea = &max_feas[batch_j * model->em_dim];
                int64_t *max_fea_index = &max_fea_indexs[batch_j * model->em_dim];
                floatx *softmax_fea = &softmax_feas[batch_j * model->category_num];

                losses[batch_j] = forward(model, train_data, text_i, max_fea, max_fea_index, softmax_fea);
                backward(model, train_data, text_i, max_fea, softmax_fea, grad_em, grad_w, grad_b);
            }

            for (int64_t batch_j = 0; batch_j < real_batch_size; batch_j++)
                s_loss += losses[batch_j];

            // 把多个batch的梯度累加起来 不可以加速，因为gt.em是临界资源
            for (int64_t batch_j = 0; batch_j < real_batch_size; batch_j++)
            {
                for (int64_t batch_k = 0; batch_k < model->em_dim * model->category_num; batch_k++)
                    gt.w[batch_k] += grads_w[batch_j * model->em_dim * model->category_num + batch_k] / (floatx)batch_size;
                for (int64_t batch_k = 0; batch_k < model->category_num; batch_k++)
                    gt.b[batch_k] += grads_b[batch_j * model->category_num + batch_k] / (floatx)batch_size;
                // em的grad 特殊对待
                for (int64_t batch_k = 0; batch_k < model->em_dim; batch_k++)
                {
                    int64_t em_index = max_fea_indexs[batch_j * model->em_dim + batch_k];
                    gt.em[em_index] += grads_em[batch_j * model->em_dim + batch_k] / (floatx)batch_size;
                }
            }

                // 计算m,v update param 可以加速
#pragma omp parallel for schedule(static) num_threads(threads_n)
            for (int64_t batch_k = 0; batch_k < model->em_dim * model->category_num; batch_k++)
            {
                adam_m.w[batch_k] = beta1 * adam_m.w[batch_k] + (1 - beta1) * gt.w[batch_k];
                adam_v.w[batch_k] = beta2 * adam_v.w[batch_k] + (1 - beta2) * gt.w[batch_k] * gt.w[batch_k];
                gt.w[batch_k] = 0.;

                floatx m_hat = adam_m.w[batch_k] / (1 - beta1t);
                floatx v_hat = adam_v.w[batch_k] / (1 - beta2t);
                model->w[batch_k] -= alpha * m_hat / ((floatx)sqrt((floatx)v_hat) + epsilon);
            }

            // 循环数量少，不用加速
            for (int64_t batch_k = 0; batch_k < model->category_num; batch_k++)
            {
                adam_m.b[batch_k] = beta1 * adam_m.b[batch_k] + (1 - beta1) * gt.b[batch_k];
                adam_v.b[batch_k] = beta2 * adam_v.b[batch_k] + (1 - beta2) * gt.b[batch_k] * gt.b[batch_k];
                gt.b[batch_k] = 0.;

                floatx m_hat = adam_m.b[batch_k] / (1 - beta1t);
                floatx v_hat = adam_v.b[batch_k] / (1 - beta2t);
                model->b[batch_k] -= alpha * m_hat / ((floatx)sqrt((floatx)v_hat) + epsilon);
            }
            // model->em是临界资源加锁
            for (int64_t batch_j = 0; batch_j < real_batch_size; batch_j++)
            {
                // em的grad 特殊对待
                for (int64_t batch_k = 0; batch_k < model->em_dim; batch_k++)
                {
                    int64_t em_index = max_fea_indexs[batch_j * model->em_dim + batch_k];
                    if (gt.em[em_index] != 0.)
                    {
                        adam_m.em[em_index] = beta1 * adam_m.em[em_index] + (1 - beta1) * gt.em[em_index];
                        adam_v.em[em_index] = beta2 * adam_v.em[em_index] + (1 - beta2) * gt.em[em_index] * gt.em[em_index];
                        gt.em[em_index] = 0.;

                        floatx m_hat = adam_m.em[em_index] / (1 - beta1t);
                        floatx v_hat = adam_v.em[em_index] / (1 - beta2t);
                        model->em[em_index] -= alpha * m_hat / ((floatx)sqrt((floatx)v_hat) + epsilon);
                    }
                }
            }

            beta1t *= beta1t;
            beta2t *= beta2t;

        } // end_batch
        epoch_end = time(NULL);
        s_loss /= train_data->text_num;
        printf("    loss: %.4f\n", s_loss);
        printf("    time: %lds\n", epoch_end - epoch_start);

        if (vali_data != NULL)
        {
            printf("evaluate vali data...\n");
            evaluate(model, vali_data, batch_size, threads_n);
        }
            
        printf("\n");

    } //end_epoch
    free(shuffle_index);
    free_model(&adam_m);
    free_model(&adam_v);
    free_model(&gt);
    free(grads_em);
    free(grads_w);
    free(grads_b);
    free(max_feas);
    free(max_fea_indexs);
    free(softmax_feas);
    free(losses);
}
void show(int64_t *a, int64_t n)
{
    for (int64_t i = 0; i < n; i++)
        printf("%ld ", a[i]);
    printf("\n");
}

int arg_helper(char *str, int argc, char **argv)
{
    int pos;
    for (pos = 1; pos < argc; pos++)
        if (strcmp(str, argv[pos]) == 0)
            return pos;
    return -1;
}
int main(int argc, char **argv)
{
    struct model_t model;
    struct dataset_t train_data, vali_data, test_data;

    int64_t em_dim = 200, vocab_num = 0, category_num = 0;
    int64_t epochs = 10, batch_size = 2000, threads_n = 20;
    floatx lr = 0.5, limit_vocab=1.;
    char *train_data_path = NULL, *vali_data_path = NULL, *test_data_path = NULL;

    int i;
    if ((i = arg_helper("-dim", argc, argv)) > 0)
        em_dim = (int64_t)atoi(argv[i + 1]);
    if ((i = arg_helper("-vocab", argc, argv)) > 0)
        vocab_num = (int64_t)atoi(argv[i + 1]);
    if ((i = arg_helper("-category", argc, argv)) > 0)
        category_num = (int64_t)atoi(argv[i + 1]);
    if ((i = arg_helper("-epoch", argc, argv)) > 0)
        epochs = (int64_t)atoi(argv[i + 1]);
    if ((i = arg_helper("-batch-size", argc, argv)) > 0)
        batch_size = (int64_t)atoi(argv[i + 1]);
    if ((i = arg_helper("-thread", argc, argv)) > 0)
        threads_n = (int64_t)atoi(argv[i + 1]);
    if ((i = arg_helper("-lr", argc, argv)) > 0)
        lr = (floatx)atof(argv[i + 1]);
    if ((i = arg_helper("-train", argc, argv)) > 0)
        train_data_path = argv[i + 1];
    if ((i = arg_helper("-vali", argc, argv)) > 0)
        vali_data_path = argv[i + 1];
    if ((i = arg_helper("-test", argc, argv)) > 0)
        test_data_path = argv[i + 1];
    if ((i = arg_helper("-limit-vocab", argc, argv)) > 0)
        limit_vocab = (floatx)atof(argv[i + 1]);

    if (vocab_num == 0) {
        printf("error: miss -vocab");
        exit(-1);
    }
    if (category_num == 0) {
        printf("error: miss -category");
        exit(-1);
    }
    if (train_data_path == NULL)
    {
        printf("error: need train data!");
        exit(-1);
    }

    init_model(&model, em_dim, vocab_num, category_num, 1);

    if (train_data_path != NULL)
        load_data(&train_data, train_data_path,(int64_t)(limit_vocab*vocab_num));
    if (test_data_path != NULL)
        load_data(&test_data, test_data_path,(int64_t)(limit_vocab*vocab_num));
    if (vali_data_path != NULL)
        load_data(&vali_data, vali_data_path,(int64_t)(limit_vocab*vocab_num));

    if (vali_data_path != NULL)
        train_adam(&model, &train_data, &vali_data, epochs, batch_size, threads_n);
    else
        train_adam(&model, &train_data, NULL, epochs, batch_size, threads_n);

    if (test_data_path != NULL)
    {
        printf("evaluate test data...\n");
        evaluate(&model, &test_data, batch_size, threads_n);
    }

    free_model(&model);
    if (train_data_path != NULL) free_data(&train_data);
    if (test_data_path != NULL) free_data(&test_data);
    if (vali_data_path != NULL) free_data(&vali_data);

    return 0;
}
