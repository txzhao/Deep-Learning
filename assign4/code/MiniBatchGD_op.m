function [RNN, sl, iter, M, min_set, eta] = MiniBatchGD_op(RNN, ...
    X, Y, n, K, m, eta, iter, epoch, M, char_to_ind, ind_to_char, smooth_loss, min_set, decay)

e = 1;
textlen = 140;
sl = [];
flag = oneHot(char_to_ind('^'), K);

while e <= length(X) - n - 1
    Xe = X(:, e : e + n - 1);
    Ye = Y(:, e + 1 : e + n);
    LIA = ismember(flag', Xe', 'rows');
    
    if e == 1 || LIA == 1
        hprev = zeros(m, 1);
    else
        hprev = h(:, end);
    end
    
    if mod(iter, 40000) == 0
        eta = eta*decay;
    end
        
    [loss, a, h, ~, p] = forward_Pass(RNN, Xe, Ye, hprev, n, K, m);
    [RNN, M] = backward_Pass(RNN, Xe, Ye, a, h, p, n, m, eta, M);
    
    if iter == 1 && e == 1
        smooth_loss = loss;
    end
    smooth_loss = 0.999*smooth_loss + 0.001*loss;
    
    if smooth_loss < min_set.loss
        min_set.RNN = RNN;
        min_set.h = hprev;
        min_set.iter = iter;
        min_set.loss = smooth_loss;
    end
    sl = [sl, smooth_loss];
%     if iter == 1 || mod(iter, 100) == 0
%         disp(['iter = ' num2str(iter) ', smooth_loss = ' num2str(smooth_loss)]);
%     end
    
    if iter == 1 || mod(iter, 10000) == 0
%         y = synText(RNN, hprev, X(:, 1), textlen, K);
%         c = [];
%         for i = 1 : textlen
%             cur_c = ind_to_char(y(i));
%             if cur_c == '^'
%                 break;
%             end
%             c = [c cur_c];
%         end
        disp('====================================================');
        disp(['epoch = ' num2str(epoch) ', iter = ' num2str(iter) ', smooth_loss = ' num2str(smooth_loss)]);
%         disp(c);
    end
    
    iter = iter + 1;
    e = e + n;
end

end


