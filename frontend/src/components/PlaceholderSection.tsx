import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';

interface PlaceholderSectionProps {
  title: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
  comingSoon?: boolean;
}

export function PlaceholderSection({ title, description, icon: Icon, comingSoon = false }: PlaceholderSectionProps) {
  return (
    <div className="p-6">
      <div className="mb-6">
        <h1>{title}</h1>
        <p className="text-muted-foreground">{description}</p>
      </div>

      <Card className="max-w-2xl mx-auto">
        <CardHeader className="text-center">
          <Icon className="w-16 h-16 mx-auto mb-4 text-muted-foreground" />
          <CardTitle>{comingSoon ? 'Coming Soon' : 'Under Development'}</CardTitle>
          <CardDescription>
            {comingSoon 
              ? 'This feature is planned for a future release'
              : 'This section is currently being built'
            }
          </CardDescription>
        </CardHeader>
        <CardContent className="text-center">
          <Button variant="outline" disabled>
            {comingSoon ? 'Stay Tuned' : 'In Progress'}
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}