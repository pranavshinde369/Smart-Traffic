import { useState } from 'react';
import { Send, Loader2, Bot } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';

export function AIChatWidget() {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleAsk = async () => {
    if (!question.trim() || loading) return;
    setLoading(true);
    setAnswer('');
    setError('');
    try {
      const res = await fetch('http://localhost:8000/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: question.trim() }),
      });
      const data = await res.json();
      setAnswer(data.answer ?? 'No response received.');
    } catch {
      setError('AI backend unreachable — ensure the server is running on port 8000.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="dashboard-card space-y-3 border-indigo-500/30">
      <div className="flex items-center gap-2">
        <div className="h-7 w-7 rounded-lg bg-indigo-500/20 flex items-center justify-center">
          <Bot className="w-4 h-4 text-indigo-400" />
        </div>
        <div>
          <h3 className="text-sm font-semibold text-foreground">Niyantran AI</h3>
          <p className="text-[10px] text-indigo-400/80">Powered by Gemini</p>
        </div>
      </div>

      <div className="flex gap-2">
        <Input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleAsk()}
          placeholder="Ask the Traffic AI..."
          className="flex-1 h-9 text-sm bg-secondary/60 border-indigo-500/20 focus-visible:ring-indigo-500/50"
        />
        <Button
          onClick={handleAsk}
          disabled={loading || !question.trim()}
          size="sm"
          className="h-9 bg-indigo-600 hover:bg-indigo-700 text-white"
        >
          {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
        </Button>
      </div>

      {(answer || error) && (
        <div className={`rounded-lg p-3 text-sm leading-relaxed ${
          error
            ? 'bg-destructive/10 border border-destructive/30 text-destructive'
            : 'bg-indigo-500/10 border border-indigo-500/20 text-foreground'
        }`}>
          {error || answer}
        </div>
      )}
    </div>
  );
}
