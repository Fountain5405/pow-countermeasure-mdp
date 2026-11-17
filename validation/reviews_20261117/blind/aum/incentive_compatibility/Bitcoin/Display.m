disp("Starting display")
disp(datetime("now"))

policyDisplayLimit = 20;

disp("Displaying upper bound policy")
disp(datetime("now"))
for i = 1:length(upperBoundPolicy)
    [a, h, fork] = stnum2st(i);
    if a <= policyDisplayLimit && h <= policyDisplayLimit
        disp([a, h, fork, upperBoundPolicy(i)])
    end
end

% disp("Displaying lower bound policy")
% disp(datetime("now"))
% for i = 1:length(lowerBoundPolicy)
%     [a, h, fork] = stnum2st(i);
%     if a <= policyDisplayLimit && h <= policyDisplayLimit
%         disp([a, h, fork, lowerBoundPolicy(i)])
%     end
% end

% disp("Displaying non-zero probability states")
% disp(datetime("now"))
% activeReceivingStates = zeros(numOfStates);
% for i=1:choices
%     for idx=1:length(P{i}.c)
%         if P{i}.P(idx) ~= 0
%             activeReceivingStates(P{i}.c(idx)) = 1;
%         end
%     end
% end
% for i = 1:numOfStates
%     if activeReceivingStates(i) ~= 0
%         [a, h, fork] = stnum2st(i);
%         if a <= policyDisplayLimit && h <= policyDisplayLimit
%             disp([a, h, fork])
%         end
%     end
% end

disp(datetime("now"))