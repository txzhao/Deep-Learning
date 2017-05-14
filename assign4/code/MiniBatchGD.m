function [RNN, smooth_loss, iter, M] = MiniBatchGD(RNN, X, Y, n, K, m, eta, iter, M, ind_to_char)

e = 1;
textlen = 200;
while e <= length(X) - n - 1
    Xe = X(:, e : e + n - 1);
    Ye = Y(:, e + 1 : e + n);
    if e == 1
        hprev = zeros(m, 1);
    else
        hprev = h(:, end);
    end
    
    [loss, a, h, ~, p] = forward_Pass(RNN, Xe, Ye, hprev, n, K, m);
    [RNN, M] = backward_Pass(RNN, Xe, Ye, a, h, p, n, m, eta, M);
    
    if e == 1
        smooth_loss = loss;
    end
    smooth_loss = 0.999*smooth_loss + 0.001*loss;
    
    % print out smoothed loss
%     if iter == 1 || mod(iter, 100) == 0
%         disp(['iter = ' num2str(iter) ', smooth_loss = ' num2str(smooth_loss)]);
%     end
    
    % print out synthesized texts
    if mod(iter, 500) == 0
        y = synText(RNN, hprev, X(:, 1), textlen, K);
        c = [];
        for i = 1 : textlen
            c = [c ind_to_char(y(i))];
        end
        disp('====================================================');
        disp(['iter = ' num2str(iter) ', smooth_loss = ' num2str(smooth_loss)]);
        disp(c);
    end
    
    iter = iter + 1;
    e = e + n;
end


